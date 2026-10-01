
if {[llength $argv] != 3} { puts stderr "usage: PRMTOP NAMDBIN OUTPDB"; exit 2 }
set prmtop [lindex $argv 0]
set coor [lindex $argv 1]
set out [lindex $argv 2]
mol new $prmtop type parm7 waitfor all
mol addfile $coor type namdbin waitfor all
set s [atomselect top all frame [expr {[molinfo top get numframes]-1}]]
$s writepdb $out
$s delete
quit
