


if {[catch {package require pbctools} pbcver]} {
    puts stderr "ERROR: PBCTools is unavailable in this VMD installation: $pbcver"
    exit 2
}

set topfile "../07_solvation_ion_validation/pVBTMA12_12Cl_015MNaCl_pilot.prmtop"
set dcdfile "05_acceptance_npt_1ns/acceptance1ns.dcd"
set outfile "analysis/vmd_box.dat"
set expected_frames 500
set polymer_sel "index 0 to 373"

if {![file exists $topfile]} {
    puts stderr "ERROR: topology missing: $topfile"
    exit 2
}
if {![file exists $dcdfile]} {
    puts stderr "ERROR: DCD missing: $dcdfile"
    exit 2
}

if {[catch {
    set molid [mol new $topfile type parm7 waitfor all]
    mol addfile $dcdfile type dcd waitfor all molid $molid
} loaderr]} {
    puts stderr "ERROR: VMD could not load Stage 08 topology/DCD: $loaderr"
    exit 2
}

set nframes [molinfo $molid get numframes]
if {$nframes != $expected_frames} {
    puts stderr "ERROR: expected $expected_frames DCD frames, VMD loaded $nframes"
    exit 2
}



set fh [open $outfile w]
puts $fh "#Frame a_A b_A c_A alpha_deg beta_deg gamma_deg"
for {set f 0} {$f < $nframes} {incr f} {
    molinfo $molid set frame $f
    set box [molinfo $molid get {a b c alpha beta gamma}]
    puts $fh "$f [join $box { }]"
}
close $fh





if {[catch {
    molinfo $molid set frame 0
    pbc join fragment -molid $molid -first 0 -last 0 -sel $polymer_sel -bondlist
    pbc wrap -molid $molid -first 0 -last 0 -sel $polymer_sel -compound fragment -center com -centersel $polymer_sel
    pbc unwrap -molid $molid -first 0 -last [expr {$nframes - 1}] -sel $polymer_sel
} pbcerr]} {
    puts stderr "ERROR: PBCTools polymer reconstruction failed: $pbcerr"
    exit 2
}

puts "VMD_PBC_STATUS PASS"
puts "VMD_FRAMES $nframes"
puts "PBCTOOLS_VERSION $pbcver"
puts "VMD_BOX_FILE $outfile"
exit 0
