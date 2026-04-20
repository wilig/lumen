//! DWARF debug info emission (lumen-v3w).
//!
//! Drives gimli to build `.debug_info` / `.debug_abbrev` / `.debug_line` /
//! `.debug_str` / `.debug_line_str` sections and attaches them to the
//! object file produced by cranelift-object. Also carries the
//! relocation machinery — debug sections reference function symbols by
//! name, and the linker resolves those references once it places the
//! functions in memory.
//!
//! MVP scope (what works after this module ships):
//!   - `bt` in gdb/lldb shows Lumen function names (via eh_frame +
//!     symbol table; already working with `unwind_info=true` set on
//!     the cranelift ISA).
//!   - `info functions` lists Lumen functions with source file +
//!     declaration line.
//!   - `break <file>.lm:<func_decl_line>` resolves.
//!
//! Not yet: per-statement line info (would need cranelift `MachSrcLoc`
//! extraction via `Context::compile` direct-compilation), variable
//! inspection, or type info. Those are cleanly additive later.
//!
//! The WriterRelocate + WriteDebugInfo patterns are lifted from
//! rustc_codegen_cranelift (same cranelift/gimli/object versions).

use cranelift_module::{DataId, FuncId};
use cranelift_object::ObjectProduct;
use gimli::write::{
    Address, AttributeValue, DwarfUnit, EndianVec, LineProgram, LineString,
    Range, RangeList, Sections, Writer,
};
use gimli::{Encoding, Format, LineEncoding, RunTimeEndian, SectionId};
use object::write::{Relocation, StandardSegment};
use object::{RelocationEncoding, RelocationFlags, SectionKind};
use std::collections::HashMap;

/// Pack a (line, col) pair into cranelift's 32-bit SourceLoc. Line
/// goes in the high 20 bits and column in the low 12. 0x0 is reserved
/// for "no info" (SourceLoc::default).
pub fn pack_srcloc(line: u32, col: u32) -> u32 {
    let l = line & 0x000F_FFFF;
    let c = col & 0x0000_0FFF;
    (l << 12) | c
}

fn unpack_srcloc(packed: u32) -> (u32, u32) {
    let line = (packed >> 12) & 0x000F_FFFF;
    let col = packed & 0x0000_0FFF;
    (line, col)
}

/// One line-program entry: a byte offset inside the function where
/// the given source line+col begins. Harvested from cranelift's
/// MachSrcLocs after compile.
#[derive(Clone, Copy)]
pub struct LineRow {
    pub offset: u32,
    pub line: u32,
    pub col: u32,
}

/// A resolved Lumen type, in the form we care about for DWARF. We
/// only distinguish the shapes gdb needs to display values; everything
/// that's a heap-allocated pointer at runtime collapses into
/// `DwarfTy::Pointer` here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DwarfTy {
    I32,
    I64,
    U32,
    U64,
    F64,
    Bool,
    Char,
    Unit,
    /// Any heap-allocated value (strings, bytes, lists, structs,
    /// sums, tuples, handles, fn ptrs — all one-word pointers in
    /// Lumen's runtime representation).
    Pointer,
}

/// A parameter captured for a function, for DW_TAG_formal_parameter.
#[derive(Clone)]
pub struct Param {
    pub name: String,
    pub ty: DwarfTy,
}

/// Per-function debug info. Populated as each Lumen fn is lowered; the
/// FuncId is the handle cranelift-object gives us and maps one-to-one
/// onto a symbol in the output ELF.
struct FunctionEntry {
    name: String,
    func_id: FuncId,
    size: u32,
    decl_line: u32,
    line_rows: Vec<LineRow>,
    /// Index into DwarfBuilder::source_files. Default 0 = the user's
    /// main source; higher indices reference imported modules.
    file_index: usize,
    params: Vec<Param>,
    ret: DwarfTy,
}

struct SourceFile {
    file_name: String,
    dir: String,
}

pub struct DwarfBuilder {
    /// Index 0 is the main source file; indices 1+ are imported-module
    /// source files registered via add_module_file.
    source_files: Vec<SourceFile>,
    /// Map of module name → index into source_files. User code has no
    /// entry (it defaults to index 0).
    module_index: HashMap<String, usize>,
    functions: Vec<FunctionEntry>,
    endian: RunTimeEndian,
    encoding: Encoding,
}

impl DwarfBuilder {
    pub fn new(source_path: &str) -> Self {
        let encoding = Encoding {
            format: Format::Dwarf32,
            version: 4,
            address_size: 8,
        };

        Self {
            source_files: vec![split_path(source_path)],
            module_index: HashMap::new(),
            functions: Vec::new(),
            endian: RunTimeEndian::Little,
            encoding,
        }
    }

    /// Register an imported module's source path so functions from it
    /// can be attributed to their real file in the debug info.
    pub fn add_module_file(&mut self, module_name: &str, source_path: &str) {
        let idx = self.source_files.len();
        self.source_files.push(split_path(source_path));
        self.module_index.insert(module_name.to_string(), idx);
    }

    /// Look up a module's file index, defaulting to 0 (main source).
    pub fn module_file_index(&self, module: Option<&str>) -> usize {
        module
            .and_then(|m| self.module_index.get(m).copied())
            .unwrap_or(0)
    }

    pub fn record_function(
        &mut self,
        name: &str,
        func_id: FuncId,
        size: u32,
        decl_line: u32,
        line_rows: Vec<LineRow>,
        file_index: usize,
        params: Vec<Param>,
        ret: DwarfTy,
    ) {
        self.functions.push(FunctionEntry {
            name: name.to_string(),
            func_id,
            size,
            decl_line,
            line_rows,
            file_index,
            params,
            ret,
        });
    }

    pub fn emit(self, product: &mut ObjectProduct) {
        if self.functions.is_empty() {
            return;
        }

        let mut dwarf = DwarfUnit::new(self.encoding);

        // Compile-unit-wide address range so debuggers know which
        // PCs belong to this object's debug info.
        let mut unit_range = RangeList(Vec::with_capacity(self.functions.len()));
        for f in &self.functions {
            unit_range.0.push(Range::StartLength {
                begin: address_for_func(f.func_id),
                length: f.size as u64,
            });
        }

        // Line program: one file per source module. User main at
        // file_id[0]; each imported module that registered a path
        // gets a higher id. DW_AT_decl_file on subprograms and
        // per-row file fields point to the right one.
        let main_file_name = LineString::new(
            self.source_files[0].file_name.as_bytes(),
            self.encoding,
            &mut dwarf.line_strings,
        );
        let comp_dir = LineString::new(
            self.source_files[0].dir.as_bytes(),
            self.encoding,
            &mut dwarf.line_strings,
        );
        let mut line_program = LineProgram::new(
            self.encoding,
            LineEncoding::default(),
            comp_dir.clone(),
            None,
            main_file_name.clone(),
            None,
        );
        let default_dir = line_program.default_directory();
        let mut file_ids: Vec<gimli::write::FileId> = Vec::with_capacity(self.source_files.len());
        file_ids.push(line_program.add_file(main_file_name, default_dir, None));
        for sf in &self.source_files[1..] {
            let dir_ls = LineString::new(sf.dir.as_bytes(), self.encoding, &mut dwarf.line_strings);
            let dir_id = line_program.add_directory(dir_ls);
            let name_ls = LineString::new(sf.file_name.as_bytes(), self.encoding, &mut dwarf.line_strings);
            file_ids.push(line_program.add_file(name_ls, dir_id, None));
        }

        for f in &self.functions {
            let file_id = file_ids[f.file_index];
            line_program.begin_sequence(Some(address_for_func(f.func_id)));
            if f.line_rows.is_empty() {
                line_program.row().file = file_id;
                line_program.row().line = f.decl_line as u64;
                line_program.row().column = 0;
                line_program.generate_row();
            } else {
                for row in &f.line_rows {
                    line_program.row().address_offset = row.offset as u64;
                    line_program.row().file = file_id;
                    line_program.row().line = row.line as u64;
                    line_program.row().column = row.col as u64;
                    line_program.row().is_statement = true;
                    line_program.generate_row();
                }
            }
            line_program.end_sequence(f.size as u64);
        }

        let range_list_id = dwarf.unit.ranges.add(unit_range);
        let name_str = dwarf.strings.add(self.source_files[0].file_name.clone());
        let dir_str = dwarf.strings.add(self.source_files[0].dir.clone());
        let producer_str = dwarf.strings.add("lumen".to_string());

        let root = dwarf.unit.root();
        let root_die = dwarf.unit.get_mut(root);
        root_die.set(gimli::DW_AT_producer, AttributeValue::StringRef(producer_str));
        root_die.set(gimli::DW_AT_language, AttributeValue::Language(gimli::DW_LANG_C));
        root_die.set(gimli::DW_AT_name, AttributeValue::StringRef(name_str));
        root_die.set(gimli::DW_AT_comp_dir, AttributeValue::StringRef(dir_str));
        root_die.set(gimli::DW_AT_ranges, AttributeValue::RangeListRef(range_list_id));
        root_die.set(gimli::DW_AT_low_pc, AttributeValue::Address(Address::Constant(0)));

        // Type DIE cache: each distinct DwarfTy gets one DIE per CU.
        let mut type_cache: HashMap<DwarfTy, gimli::write::UnitEntryId> = HashMap::new();

        // One DW_TAG_subprogram DIE per Lumen function. Attributes
        // reference the source file it came from and its parameter /
        // return types; nested DW_TAG_formal_parameter DIEs name
        // each arg. DW_AT_location is deliberately absent for MVP —
        // gdb reports "optimized out" for values but `info args`
        // shows signatures correctly.
        for f in &self.functions {
            let file_id = file_ids[f.file_index];
            let ret_ty_id = if f.ret == DwarfTy::Unit {
                None
            } else {
                Some(get_or_build_type(&mut dwarf, root, &mut type_cache, &f.ret))
            };
            let subp = dwarf.unit.add(root, gimli::DW_TAG_subprogram);
            let fname = dwarf.strings.add(f.name.clone());
            let die = dwarf.unit.get_mut(subp);
            die.set(gimli::DW_AT_name, AttributeValue::StringRef(fname));
            die.set(gimli::DW_AT_low_pc, AttributeValue::Address(address_for_func(f.func_id)));
            die.set(gimli::DW_AT_high_pc, AttributeValue::Udata(f.size as u64));
            die.set(gimli::DW_AT_decl_file, AttributeValue::FileIndex(Some(file_id)));
            die.set(gimli::DW_AT_decl_line, AttributeValue::Udata(f.decl_line as u64));
            die.set(gimli::DW_AT_external, AttributeValue::Flag(true));
            if let Some(rt) = ret_ty_id {
                die.set(gimli::DW_AT_type, AttributeValue::UnitRef(rt));
            }

            // Formal parameters as children of the subprogram DIE.
            // DW_AT_location uses the System V AMD64 ABI register
            // assignment — int/ptr params in rdi/rsi/rdx/rcx/r8/r9,
            // float params in xmm0..xmm7. Accurate at function entry;
            // gdb may show stale values after the fn moves registers
            // to stack slots (full range info is a follow-up).
            let param_ty_ids: Vec<_> = f.params.iter()
                .map(|p| get_or_build_type(&mut dwarf, root, &mut type_cache, &p.ty))
                .collect();
            let mut int_slot: usize = 0;
            let mut float_slot: usize = 0;
            // DWARF 5, section 3.4: x86-64 register numbering. Int
            // ABI regs in order: rdi=5, rsi=4, rdx=1, rcx=2, r8=8, r9=9.
            // Float ABI regs: xmm0=17 ... xmm7=24.
            const INT_DWARF_REGS: [u8; 6] = [5, 4, 1, 2, 8, 9];
            const FLOAT_DWARF_REGS: [u8; 8] = [17, 18, 19, 20, 21, 22, 23, 24];
            for (p, &tid) in f.params.iter().zip(param_ty_ids.iter()) {
                let param_die = dwarf.unit.add(subp, gimli::DW_TAG_formal_parameter);
                let pname = dwarf.strings.add(p.name.clone());
                let reg = if matches!(p.ty, DwarfTy::F64) {
                    let r = FLOAT_DWARF_REGS.get(float_slot).copied();
                    float_slot += 1;
                    r
                } else {
                    let r = INT_DWARF_REGS.get(int_slot).copied();
                    int_slot += 1;
                    r
                };
                let d = dwarf.unit.get_mut(param_die);
                d.set(gimli::DW_AT_name, AttributeValue::StringRef(pname));
                d.set(gimli::DW_AT_type, AttributeValue::UnitRef(tid));
                if let Some(r) = reg {
                    // DW_OP_reg0..DW_OP_reg31 are single-byte opcodes
                    // 0x50..0x6f. For regs >=32 we'd need DW_OP_regx
                    // + ULEB — none of the ABI regs trigger that.
                    let mut expr = gimli::write::Expression::new();
                    expr.op_reg(gimli::Register(r as u16));
                    d.set(gimli::DW_AT_location, AttributeValue::Exprloc(expr));
                }
            }
        }

        dwarf.unit.line_program = line_program;

        // Emit all sections via our relocation-capturing writer.
        let mut sections = Sections::new(WriterRelocate::new(self.endian));
        dwarf.write(&mut sections).unwrap();

        // Two-pass: first add sections so gimli-section-to-object-section
        // mapping is available when we apply relocs.
        let mut section_map: HashMap<SectionId, (object::write::SectionId, object::write::SymbolId)> = HashMap::new();
        let _: gimli::write::Result<()> = sections.for_each_mut(|id, section: &mut WriterRelocate| {
            if section.writer.slice().is_empty() {
                return Ok(());
            }
            let obj_section = add_debug_section(&mut product.object, id, section.writer.take());
            section_map.insert(id, obj_section);
            Ok(())
        });

        // Collect relocs first so we don't borrow both `sections` and
        // `product` simultaneously in the second pass.
        let mut pending: Vec<((object::write::SectionId, object::write::SymbolId), DebugReloc)> = Vec::new();
        let _: gimli::write::Result<()> = sections.for_each(|id, section: &WriterRelocate| {
            if let Some(&from) = section_map.get(&id) {
                for reloc in &section.relocs {
                    pending.push((from, reloc.clone()));
                }
            }
            Ok(())
        });
        // Snapshot the symbol maps so the reloc pass can borrow
        // `product.object` mutably without conflicting.
        let mut func_map: HashMap<u32, object::write::SymbolId> = HashMap::new();
        for (fid, slot) in product.functions.iter() {
            if let Some((sym, _)) = slot {
                func_map.insert(fid.as_u32(), *sym);
            }
        }
        let mut data_map: HashMap<u32, object::write::SymbolId> = HashMap::new();
        for (did, slot) in product.data_objects.iter() {
            if let Some((sym, _)) = slot {
                data_map.insert(did.as_u32(), *sym);
            }
        }

        for (from, reloc) in pending {
            add_debug_reloc(&mut product.object, &section_map, &func_map, &data_map, from, &reloc);
        }
    }
}

// --- WriterRelocate: a gimli::write::Writer that captures cross-section
// and cross-symbol references as relocations rather than trying to
// resolve them eagerly. Pattern from rustc_codegen_cranelift.

#[derive(Clone)]
struct DebugReloc {
    offset: u32,
    size: u8,
    name: DebugRelocName,
    addend: i64,
    kind: object::RelocationKind,
}

#[derive(Clone)]
enum DebugRelocName {
    Section(SectionId),
    Symbol(usize),
}

#[derive(Clone)]
struct WriterRelocate {
    relocs: Vec<DebugReloc>,
    writer: EndianVec<RunTimeEndian>,
}

impl WriterRelocate {
    fn new(endian: RunTimeEndian) -> Self {
        Self { relocs: Vec::new(), writer: EndianVec::new(endian) }
    }
}

impl Writer for WriterRelocate {
    type Endian = RunTimeEndian;

    fn endian(&self) -> Self::Endian { self.writer.endian() }
    fn len(&self) -> usize { self.writer.len() }
    fn write(&mut self, bytes: &[u8]) -> gimli::write::Result<()> { self.writer.write(bytes) }
    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> gimli::write::Result<()> {
        self.writer.write_at(offset, bytes)
    }

    fn write_address(&mut self, address: Address, size: u8) -> gimli::write::Result<()> {
        match address {
            Address::Constant(val) => self.write_udata(val, size),
            Address::Symbol { symbol, addend } => {
                let offset = self.len() as u32;
                self.relocs.push(DebugReloc {
                    offset,
                    size,
                    name: DebugRelocName::Symbol(symbol),
                    addend,
                    kind: object::RelocationKind::Absolute,
                });
                self.write_udata(0, size)
            }
        }
    }

    fn write_offset(&mut self, val: usize, section: SectionId, size: u8) -> gimli::write::Result<()> {
        let offset = self.len() as u32;
        self.relocs.push(DebugReloc {
            offset,
            size,
            name: DebugRelocName::Section(section),
            addend: val as i64,
            kind: object::RelocationKind::Absolute,
        });
        self.write_udata(0, size)
    }

    fn write_offset_at(
        &mut self,
        offset: usize,
        val: usize,
        section: SectionId,
        size: u8,
    ) -> gimli::write::Result<()> {
        self.relocs.push(DebugReloc {
            offset: offset as u32,
            size,
            name: DebugRelocName::Section(section),
            addend: val as i64,
            kind: object::RelocationKind::Absolute,
        });
        self.write_udata_at(offset, 0, size)
    }

    fn write_eh_pointer(
        &mut self,
        address: Address,
        eh_pe: gimli::DwEhPe,
        size: u8,
    ) -> gimli::write::Result<()> {
        match address {
            Address::Constant(val) => {
                let val = match eh_pe.application() {
                    gimli::DW_EH_PE_absptr => val,
                    gimli::DW_EH_PE_pcrel => {
                        let offset = self.len() as u64;
                        offset.wrapping_sub(val)
                    }
                    _ => return Err(gimli::write::Error::UnsupportedPointerEncoding(eh_pe)),
                };
                self.write_eh_pointer_data(val, eh_pe.format(), size)
            }
            Address::Symbol { symbol, addend } => match eh_pe.application() {
                gimli::DW_EH_PE_pcrel => {
                    let sz = match eh_pe.format() {
                        gimli::DW_EH_PE_sdata4 => 4,
                        gimli::DW_EH_PE_sdata8 => 8,
                        _ => return Err(gimli::write::Error::UnsupportedPointerEncoding(eh_pe)),
                    };
                    self.relocs.push(DebugReloc {
                        offset: self.len() as u32,
                        size: sz,
                        name: DebugRelocName::Symbol(symbol),
                        addend,
                        kind: object::RelocationKind::Relative,
                    });
                    self.write_udata(0, sz)
                }
                gimli::DW_EH_PE_absptr => {
                    self.relocs.push(DebugReloc {
                        offset: self.len() as u32,
                        size,
                        name: DebugRelocName::Symbol(symbol),
                        addend,
                        kind: object::RelocationKind::Absolute,
                    });
                    self.write_udata(0, size)
                }
                _ => Err(gimli::write::Error::UnsupportedPointerEncoding(eh_pe)),
            },
        }
    }
}

// --- Object file integration ------------------------------------------------

fn add_debug_section(
    obj: &mut object::write::Object<'_>,
    id: SectionId,
    data: Vec<u8>,
) -> (object::write::SectionId, object::write::SymbolId) {
    let name = if obj.format() == object::BinaryFormat::MachO {
        id.name().replace('.', "__")
    } else {
        id.name().to_string()
    }
    .into_bytes();

    let segment = obj.segment_name(StandardSegment::Debug).to_vec();
    let section_id = obj.add_section(
        segment,
        name,
        if id == SectionId::DebugStr || id == SectionId::DebugLineStr {
            SectionKind::DebugString
        } else if id == SectionId::EhFrame {
            SectionKind::ReadOnlyData
        } else {
            SectionKind::Debug
        },
    );
    obj.section_mut(section_id)
        .set_data(data, if id == SectionId::EhFrame { 8 } else { 1 });
    let sym = obj.section_symbol(section_id);
    (section_id, sym)
}

fn add_debug_reloc(
    obj: &mut object::write::Object<'_>,
    section_map: &HashMap<SectionId, (object::write::SectionId, object::write::SymbolId)>,
    func_map: &HashMap<u32, object::write::SymbolId>,
    data_map: &HashMap<u32, object::write::SymbolId>,
    from: (object::write::SectionId, object::write::SymbolId),
    reloc: &DebugReloc,
) {
    let (symbol, symbol_offset) = match reloc.name {
        DebugRelocName::Section(id) => (section_map.get(&id).unwrap().1, 0),
        DebugRelocName::Symbol(raw) => {
            let raw = raw as u32;
            let symbol_id = if raw & (1 << 31) == 0 {
                *func_map.get(&raw).expect("unknown function symbol in DWARF reloc")
            } else {
                *data_map.get(&(raw & !(1 << 31))).expect("unknown data symbol in DWARF reloc")
            };
            obj.symbol_section_and_offset(symbol_id).unwrap_or((symbol_id, 0))
        }
    };
    obj.add_relocation(
        from.0,
        Relocation {
            offset: u64::from(reloc.offset),
            symbol,
            flags: RelocationFlags::Generic {
                kind: reloc.kind,
                encoding: RelocationEncoding::Generic,
                size: reloc.size * 8,
            },
            addend: i64::try_from(symbol_offset).unwrap() + reloc.addend,
        },
    )
    .unwrap();
}

/// Look up or build a DIE for a Lumen type, caching per CU so each
/// shape is defined exactly once. Base types go directly under the
/// compile unit root; `DwarfTy::Pointer` becomes a DW_TAG_pointer_type
/// over a DW_TAG_unspecified_type (opaque `void *` — good enough for
/// MVP, individual struct layouts are a follow-up ticket).
fn get_or_build_type(
    dwarf: &mut DwarfUnit,
    root: gimli::write::UnitEntryId,
    cache: &mut HashMap<DwarfTy, gimli::write::UnitEntryId>,
    ty: &DwarfTy,
) -> gimli::write::UnitEntryId {
    if let Some(&id) = cache.get(ty) {
        return id;
    }
    let id = match ty {
        DwarfTy::I32 => build_base_type(dwarf, root, "i32", gimli::DW_ATE_signed, 4),
        DwarfTy::I64 => build_base_type(dwarf, root, "i64", gimli::DW_ATE_signed, 8),
        DwarfTy::U32 => build_base_type(dwarf, root, "u32", gimli::DW_ATE_unsigned, 4),
        DwarfTy::U64 => build_base_type(dwarf, root, "u64", gimli::DW_ATE_unsigned, 8),
        DwarfTy::F64 => build_base_type(dwarf, root, "f64", gimli::DW_ATE_float, 8),
        DwarfTy::Bool => build_base_type(dwarf, root, "bool", gimli::DW_ATE_boolean, 4),
        DwarfTy::Char => build_base_type(dwarf, root, "char", gimli::DW_ATE_UTF, 4),
        DwarfTy::Unit => build_base_type(dwarf, root, "unit", gimli::DW_ATE_unsigned, 4),
        DwarfTy::Pointer => {
            let base = dwarf.unit.add(root, gimli::DW_TAG_unspecified_type);
            let bname = dwarf.strings.add("<lumen>".to_string());
            dwarf.unit.get_mut(base).set(gimli::DW_AT_name, AttributeValue::StringRef(bname));
            let ptr = dwarf.unit.add(root, gimli::DW_TAG_pointer_type);
            dwarf.unit.get_mut(ptr).set(gimli::DW_AT_byte_size, AttributeValue::Udata(8));
            dwarf.unit.get_mut(ptr).set(gimli::DW_AT_type, AttributeValue::UnitRef(base));
            ptr
        }
    };
    cache.insert(ty.clone(), id);
    id
}

fn build_base_type(
    dwarf: &mut DwarfUnit,
    root: gimli::write::UnitEntryId,
    name: &str,
    encoding: gimli::DwAte,
    size: u64,
) -> gimli::write::UnitEntryId {
    let id = dwarf.unit.add(root, gimli::DW_TAG_base_type);
    let name_str = dwarf.strings.add(name.to_string());
    let die = dwarf.unit.get_mut(id);
    die.set(gimli::DW_AT_name, AttributeValue::StringRef(name_str));
    die.set(gimli::DW_AT_encoding, AttributeValue::Encoding(encoding));
    die.set(gimli::DW_AT_byte_size, AttributeValue::Udata(size));
    id
}

/// Split a source path into (file_name, directory). Canonicalizes if
/// possible so relative paths still give gdb an absolute dir to resolve.
fn split_path(path: &str) -> SourceFile {
    let absolute = std::fs::canonicalize(path).unwrap_or_else(|_| path.into());
    let file_name = absolute
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    let dir = absolute
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    SourceFile { file_name, dir }
}

/// Encode a FuncId as the `symbol` payload of `Address::Symbol`. Matches
/// the trick rustc_codegen_cranelift uses — high bit 0 for function,
/// high bit 1 for data. We only use the function form today.
fn address_for_func(func_id: FuncId) -> Address {
    let symbol = func_id.as_u32();
    assert!(symbol & (1 << 31) == 0);
    Address::Symbol {
        symbol: symbol as usize,
        addend: 0,
    }
}
