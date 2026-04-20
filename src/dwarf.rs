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
/// distinguish the shapes gdb needs to display values; most heap shapes
/// collapse into `DwarfTy::Pointer`, but named user structs get their
/// own variant so we can attach field layouts to them.
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
    /// Pointer to a registered user struct (see DwarfBuilder::add_struct).
    /// Renders as `<Name> *` in gdb and `print *p` displays fields.
    Struct(String),
    /// Any heap-allocated value not otherwise classified (strings,
    /// bytes, lists, maps, sums, tuples, handles, fn ptrs — all
    /// one-word pointers in Lumen's runtime representation).
    Pointer,
}

/// A field inside a user struct layout.
#[derive(Clone)]
pub struct StructField {
    pub name: String,
    pub ty: DwarfTy,
    /// Byte offset of this field from the start of the struct's
    /// payload (the pointer gdb sees). Lumen's rc header (8 bytes) is
    /// BEFORE the payload, so offsets are 0-based here.
    pub offset: u32,
}

/// One entry in a DWARF location list: over the byte-offset range
/// [start, end) within the function, the value lives at `loc`.
#[derive(Clone, Copy, Debug)]
pub enum VarLoc {
    /// Hardware register, indexed by its DWARF register number.
    Reg(u16),
    /// Byte offset from the Canonical Frame Address (`DW_OP_call_frame_cfa`
    /// + a signed integer).
    CfaOffset(i64),
}

#[derive(Clone, Debug)]
pub struct LocRange {
    pub start: u32,
    pub end: u32,
    pub loc: VarLoc,
}

/// A parameter captured for a function, for DW_TAG_formal_parameter.
#[derive(Clone)]
pub struct Param {
    pub name: String,
    pub ty: DwarfTy,
    /// Empty → fall back to ABI-based single-register location (from
    /// ax3). Non-empty → emit a .debug_loc location list so gdb can
    /// track the value across spills and moves.
    pub locations: Vec<LocRange>,
}

/// A local binding (let/var/for-binder/match-binding) that lives in a
/// function's scope. Emitted as DW_TAG_variable inside the subprogram.
#[derive(Clone)]
pub struct Local {
    pub name: String,
    pub ty: DwarfTy,
    pub locations: Vec<LocRange>,
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
    locals: Vec<Local>,
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
    /// Registered struct layouts keyed by struct name. Consumed at
    /// emit-time to build DW_TAG_structure_type DIEs.
    structs: HashMap<String, (Vec<StructField>, u32)>,
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
            structs: HashMap::new(),
            endian: RunTimeEndian::Little,
            encoding,
        }
    }

    /// Register a user-struct layout. `byte_size` is the total payload
    /// size; the rc header lives before the pointer we give gdb, so
    /// offsets passed here are 0-based from the payload start.
    pub fn add_struct(&mut self, name: &str, fields: Vec<StructField>, byte_size: u32) {
        self.structs.insert(name.to_string(), (fields, byte_size));
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
        locals: Vec<Local>,
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
            locals,
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

        // Eagerly emit DIEs for every registered struct so gdb can
        // resolve them even if no fn param/return directly references
        // the struct (the common case — most structs flow through
        // generic Pointer-typed params).
        let struct_names: Vec<String> = self.structs.keys().cloned().collect();
        for name in struct_names {
            get_or_build_type(&mut dwarf, root, &mut type_cache, &self.structs, &DwarfTy::Struct(name));
        }

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
                Some(get_or_build_type(&mut dwarf, root, &mut type_cache, &self.structs, &f.ret))
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

            // Formal parameters: DW_AT_location comes from cranelift's
            // value_labels_ranges if the binding was tracked, else
            // falls back to the System V AMD64 ABI register guess
            // (correct at function entry only).
            let param_ty_ids: Vec<_> = f.params.iter()
                .map(|p| get_or_build_type(&mut dwarf, root, &mut type_cache, &self.structs, &p.ty))
                .collect();
            let mut int_slot: usize = 0;
            let mut float_slot: usize = 0;
            const INT_DWARF_REGS: [u8; 6] = [5, 4, 1, 2, 8, 9];
            const FLOAT_DWARF_REGS: [u8; 8] = [17, 18, 19, 20, 21, 22, 23, 24];
            for (p, &tid) in f.params.iter().zip(param_ty_ids.iter()) {
                let param_die = dwarf.unit.add(subp, gimli::DW_TAG_formal_parameter);
                let pname = dwarf.strings.add(p.name.clone());
                // Track the ABI slot regardless of whether we end up
                // using it — so slot counts stay aligned with the
                // Lumen param order.
                let abi_reg = if matches!(p.ty, DwarfTy::F64) {
                    let r = FLOAT_DWARF_REGS.get(float_slot).copied();
                    float_slot += 1;
                    r
                } else {
                    let r = INT_DWARF_REGS.get(int_slot).copied();
                    int_slot += 1;
                    r
                };
                {
                    let d = dwarf.unit.get_mut(param_die);
                    d.set(gimli::DW_AT_name, AttributeValue::StringRef(pname));
                    d.set(gimli::DW_AT_type, AttributeValue::UnitRef(tid));
                }
                if !p.locations.is_empty() {
                    set_location_from_ranges(&mut dwarf, param_die, f.func_id, f.size, &p.locations);
                } else if let Some(r) = abi_reg {
                    let mut expr = gimli::write::Expression::new();
                    expr.op_reg(gimli::Register(r as u16));
                    dwarf.unit.get_mut(param_die).set(
                        gimli::DW_AT_location,
                        AttributeValue::Exprloc(expr),
                    );
                }
            }

            // Locals as DW_TAG_variable children. Emitted when
            // cranelift tracked their locations; otherwise we still
            // emit the DIE (name + type) so `info locals` shows them,
            // but gdb says "optimized out" for the value.
            let local_ty_ids: Vec<_> = f.locals.iter()
                .map(|l| get_or_build_type(&mut dwarf, root, &mut type_cache, &self.structs, &l.ty))
                .collect();
            for (l, &tid) in f.locals.iter().zip(local_ty_ids.iter()) {
                let var_die = dwarf.unit.add(subp, gimli::DW_TAG_variable);
                let lname = dwarf.strings.add(l.name.clone());
                {
                    let d = dwarf.unit.get_mut(var_die);
                    d.set(gimli::DW_AT_name, AttributeValue::StringRef(lname));
                    d.set(gimli::DW_AT_type, AttributeValue::UnitRef(tid));
                }
                if !l.locations.is_empty() {
                    set_location_from_ranges(&mut dwarf, var_die, f.func_id, f.size, &l.locations);
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
/// compile unit root. `DwarfTy::Struct(name)` emits a
/// DW_TAG_structure_type with DW_TAG_member children (if the struct
/// was registered via add_struct) and returns a pointer-to-struct DIE.
/// `DwarfTy::Pointer` is a generic opaque `void *` for shapes whose
/// layout we don't yet describe (sums, tuples, lists, maps).
fn get_or_build_type(
    dwarf: &mut DwarfUnit,
    root: gimli::write::UnitEntryId,
    cache: &mut HashMap<DwarfTy, gimli::write::UnitEntryId>,
    structs: &HashMap<String, (Vec<StructField>, u32)>,
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
        DwarfTy::Struct(name) => build_struct_pointer_type(dwarf, root, cache, structs, name),
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

/// Build a DW_TAG_structure_type for a registered user struct (with
/// DW_TAG_member children for each field) and wrap it in a
/// DW_TAG_pointer_type — Lumen hands gdb a payload pointer, so the
/// type users see is a pointer-to-struct.
fn build_struct_pointer_type(
    dwarf: &mut DwarfUnit,
    root: gimli::write::UnitEntryId,
    cache: &mut HashMap<DwarfTy, gimli::write::UnitEntryId>,
    structs: &HashMap<String, (Vec<StructField>, u32)>,
    name: &str,
) -> gimli::write::UnitEntryId {
    // Build the structure body.
    let struct_id = dwarf.unit.add(root, gimli::DW_TAG_structure_type);
    let name_str = dwarf.strings.add(name.to_string());
    {
        let die = dwarf.unit.get_mut(struct_id);
        die.set(gimli::DW_AT_name, AttributeValue::StringRef(name_str));
    }
    if let Some((fields, byte_size)) = structs.get(name) {
        dwarf.unit.get_mut(struct_id).set(
            gimli::DW_AT_byte_size,
            AttributeValue::Udata(*byte_size as u64),
        );
        // Collect field types first so the recursive type build
        // doesn't conflict with our mutable borrow of struct_id.
        let field_types: Vec<_> = fields.iter()
            .map(|f| (f.name.clone(), f.offset, get_or_build_type(dwarf, root, cache, structs, &f.ty)))
            .collect();
        for (fname, offset, fty_id) in field_types {
            let mem = dwarf.unit.add(struct_id, gimli::DW_TAG_member);
            let fname_str = dwarf.strings.add(fname);
            let d = dwarf.unit.get_mut(mem);
            d.set(gimli::DW_AT_name, AttributeValue::StringRef(fname_str));
            d.set(gimli::DW_AT_type, AttributeValue::UnitRef(fty_id));
            d.set(gimli::DW_AT_data_member_location, AttributeValue::Udata(offset as u64));
        }
    }
    // Wrap in a pointer so the param/local's declared type matches
    // what the Lumen runtime actually hands gdb (a payload ptr).
    let ptr_id = dwarf.unit.add(root, gimli::DW_TAG_pointer_type);
    dwarf.unit.get_mut(ptr_id).set(gimli::DW_AT_byte_size, AttributeValue::Udata(8));
    dwarf.unit.get_mut(ptr_id).set(gimli::DW_AT_type, AttributeValue::UnitRef(struct_id));
    ptr_id
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

/// Translate a `VarLoc` to a gimli DWARF expression.
fn varloc_to_expr(loc: VarLoc) -> gimli::write::Expression {
    let mut expr = gimli::write::Expression::new();
    match loc {
        VarLoc::Reg(n) => expr.op_reg(gimli::Register(n)),
        VarLoc::CfaOffset(off) => {
            // `DW_OP_call_frame_cfa` + constant + `DW_OP_plus`.
            expr.op(gimli::DW_OP_call_frame_cfa);
            expr.op_consts(off);
            expr.op(gimli::DW_OP_plus);
        }
    }
    expr
}

/// Attach DW_AT_location to a DIE. Single-range bindings that span
/// the whole function become a DW_FORM_exprloc; multi-range or
/// partial ones become a .debug_loc location list so gdb can
/// track the value as cranelift moves it between regs and spills.
fn set_location_from_ranges(
    dwarf: &mut DwarfUnit,
    die_id: gimli::write::UnitEntryId,
    func_id: FuncId,
    func_size: u32,
    ranges: &[LocRange],
) {
    let covers_whole = ranges.len() == 1
        && ranges[0].start == 0
        && ranges[0].end >= func_size;
    if covers_whole {
        dwarf.unit.get_mut(die_id).set(
            gimli::DW_AT_location,
            AttributeValue::Exprloc(varloc_to_expr(ranges[0].loc)),
        );
        return;
    }
    let mut list = gimli::write::LocationList(Vec::with_capacity(ranges.len()));
    for r in ranges {
        list.0.push(gimli::write::Location::StartEnd {
            begin: Address::Symbol {
                symbol: func_id.as_u32() as usize,
                addend: r.start as i64,
            },
            end: Address::Symbol {
                symbol: func_id.as_u32() as usize,
                addend: r.end as i64,
            },
            data: varloc_to_expr(r.loc),
        });
    }
    let list_id = dwarf.unit.locations.add(list);
    dwarf.unit.get_mut(die_id).set(
        gimli::DW_AT_location,
        AttributeValue::LocationListRef(list_id),
    );
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
