if {[catch {package require pbctools} pbcver]} { puts stderr "ERROR: PBCTools unavailable: $pbcver"; exit 2 }
set topfile "build/pVBTMA12_PFOA_assoc.prmtop"
set dcdfile "03_acceptance_npt_1ns/acceptance1ns.dcd"
set outfile "analysis/vmd_box.dat"
set expected_frames 500
if {[catch {set m [mol new $topfile type parm7 waitfor all]; mol addfile $dcdfile type dcd waitfor all molid $m} err]} {puts stderr "ERROR: load failed: $err"; exit 2}
set n [molinfo $m get numframes]
if {$n != $expected_frames} {puts stderr "ERROR: expected $expected_frames frames, got $n"; exit 2}
set fh [open $outfile w]; puts $fh "#Frame a_A b_A c_A alpha beta gamma"
for {set f 0} {$f<$n} {incr f} {molinfo $m set frame $f; puts $fh "$f [join [molinfo $m get {a b c alpha beta gamma}] { }]"} ; close $fh
if {[catch {pbc join fragment -molid $m -first 0 -last 0 -sel "resname PVB" -bondlist; pbc wrap -molid $m -first 0 -last 0 -sel "resname PVB PFO" -compound fragment -center com -centersel "resname PVB"; pbc unwrap -molid $m -first 0 -last [expr {$n-1}] -sel "resname PVB PFO"} e]} {puts stderr "ERROR: PBCTools reconstruction failed: $e"; exit 2}
puts "VMD_PBC_STATUS PASS"; puts "VMD_FRAMES $n"; puts "PBCTOOLS_VERSION $pbcver"; exit 0
