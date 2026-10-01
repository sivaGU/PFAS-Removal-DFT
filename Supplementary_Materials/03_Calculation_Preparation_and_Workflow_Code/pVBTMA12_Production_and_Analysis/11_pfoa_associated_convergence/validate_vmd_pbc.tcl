
if {[catch {package require pbctools} pbcver]} { puts stderr "ERROR: PBCTools unavailable: $pbcver"; exit 2 }
set stage10 "../10_pfoa_associated_production"
set fhman [open "analysis/analysis_manifest.json" r]; set manifest [read $fhman]; close $fhman
regexp {"chunks_per_replica"[[:space:]]*:[[:space:]]*([0-9]+)} $manifest -> chunks
for {set r 1} {$r <= 3} {incr r} {
  set rr [format "%02d" $r]; set topfile "$stage10/inputs/pVBTMA12_PFOA_assoc.prmtop"
  if {[catch {set m [mol new $topfile type parm7 waitfor all]} err]} { puts stderr "ERROR: topology load failed: $err"; exit 2 }
  for {set c 1} {$c <= $chunks} {incr c} { set cc [format "%02d" $c]; set dcd "$stage10/replica_$rr/${cc}_prod_1ns/prod_$cc.dcd"; if {[catch {mol addfile $dcd type dcd waitfor all molid $m} err]} { puts stderr "ERROR: DCD load failed: $err"; exit 2 } }
  set expected [expr {$chunks * 500}]; set n [molinfo $m get numframes]; if {$n != $expected} { puts stderr "ERROR: replica $rr expected $expected frames, got $n"; exit 2 }
  set fh [open "analysis/replica_$rr/vmd_box.dat" w]; puts $fh "#Frame a_A b_A c_A alpha beta gamma"
  for {set f 0} {$f < $n} {incr f} { molinfo $m set frame $f; puts $fh "$f [join [molinfo $m get {a b c alpha beta gamma}] { }]" }; close $fh
  if {[catch {pbc join fragment -molid $m -first 0 -last 0 -sel "resname PVB" -bondlist; pbc wrap -molid $m -first 0 -last 0 -sel "resname PVB PFO" -compound fragment -center com -centersel "resname PVB"; pbc unwrap -molid $m -first 0 -last [expr {$n-1}] -sel "resname PVB PFO"} e]} { puts stderr "ERROR: replica $rr PBCTools reconstruction failed: $e"; exit 2 }
  mol delete $m; puts "VMD_PBC_REPLICA_$rr PASS frames=$n"
}
puts "VMD_PBC_STATUS PASS"; puts "PBCTOOLS_VERSION $pbcver"; exit 0
