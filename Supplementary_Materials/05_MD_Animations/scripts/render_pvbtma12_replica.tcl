if {[llength $argv] != 8} {
    puts stderr "FATAL: expected topology replica_dir frame_dir replica_id stride frame_limit width height"
    exit 2
}

lassign $argv topology replica_dir frame_dir replica_id stride frame_limit width height
if {$stride < 1 || $frame_limit < 1} {
    puts stderr "FATAL: stride and frame_limit must be positive"
    exit 2
}
file mkdir $frame_dir

set molid [mol new $topology type parm7 waitfor all]
set source_frames {}
for {set chunk 1} {$chunk <= 10} {incr chunk} {
    set subdir [format "%02d_prod_1ns" $chunk]
    set dcd [file join $replica_dir $subdir [format "prod_%02d.dcd" $chunk]]
    if {![file isfile $dcd]} {
        puts stderr "FATAL: missing $dcd"
        exit 2
    }
    set before [molinfo $molid get numframes]
    mol addfile $dcd type dcd first 0 last -1 step $stride waitfor all molid $molid
    set added [expr {[molinfo $molid get numframes] - $before}]
    if {$added != [expr {int(ceil(500.0 / $stride))}]} {
        puts stderr "FATAL: $dcd contributed $added sampled frames"
        exit 2
    }
    for {set i 0} {$i < $added} {incr i} {
        lappend source_frames [list $chunk [expr {$i * $stride}]]
    }
}
set final_dcd [file join $replica_dir 10_prod_1ns prod_10.dcd]
mol addfile $final_dcd type dcd first 499 last 499 waitfor all molid $molid
lappend source_frames [list 10 499]

if {[molinfo $molid get numframes] != [llength $source_frames]} {
    puts stderr "FATAL: trajectory/frame manifest mismatch"
    exit 2
}
if {[catch {package require pbctools} err]} {
    puts stderr "FATAL: PBCTools unavailable: $err"
    exit 2
}
if {[catch {pbc wrap -molid $molid -all -compound residue -center com -centersel "resname PVB"} err]} {
    puts stderr "FATAL: PBC imaging failed: $err"
    exit 2
}

mol delrep 0 $molid
display projection Orthographic
display depthcue off
display ambientocclusion off
display shadows off
axes location Off
color Display Background white

mol representation Licorice 0.16 12 12
mol color Element
mol selection "resname PVB and not hydrogen"
mol material Opaque
mol addrep $molid

mol representation Licorice 0.24 14 14
mol color Element
mol selection "resname PFO and not hydrogen"
mol material Opaque
mol addrep $molid

mol representation VDW 0.42 12
mol color ColorID 7
mol selection "element Cl and within 10 of (resname PVB or resname PFO)"
mol material Opaque
mol addrep $molid
mol selupdate 2 $molid on

mol representation VDW 0.34 12
mol color ColorID 4
mol selection "element Na and within 10 of (resname PVB or resname PFO)"
mol material Opaque
mol addrep $molid
mol selupdate 3 $molid on

set reference [atomselect $molid "resname PVB and not hydrogen" frame 0]
set manifest [open [file join $frame_dir frame_manifest.csv] w]
puts $manifest "movie_frame,chunk,source_frame,time_ns"
set subtitles [open [file join $frame_dir frame_labels.srt] w]
set count [llength $source_frames]
set render_indices {}
if {$frame_limit < $count} {
    foreach index {0 9 10 50 99 100} {
        if {$index < $count} {lappend render_indices $index}
    }
} else {
    for {set index 0} {$index < $count} {incr index} {
        lappend render_indices $index
    }
}
set count [llength $render_indices]

proc srt_time {milliseconds} {
    set hours [expr {int($milliseconds / 3600000)}]
    set minutes [expr {int(($milliseconds % 3600000) / 60000)}]
    set seconds [expr {int(($milliseconds % 60000) / 1000)}]
    set millis [expr {int($milliseconds % 1000)}]
    return [format "%02d:%02d:%02d,%03d" $hours $minutes $seconds $millis]
}

for {set movie_frame 0} {$movie_frame < $count} {incr movie_frame} {
    set source_index [lindex $render_indices $movie_frame]
    animate goto $source_index
    lassign [lindex $source_frames $source_index] chunk source_frame
    set mobile [atomselect $molid "resname PVB and not hydrogen" frame $source_index]
    set all [atomselect $molid all frame $source_index]
    $all move [measure fit $mobile $reference]
    set focus [atomselect $molid "resname PVB or resname PFO" frame $source_index]
    set center [measure center $focus]
    $all moveby [vecscale -1 $center]
    $focus delete
    $mobile delete
    $all delete

    if {$movie_frame == 0} {
        display resetview
        rotate x by -65
        rotate y by 15
        rotate z by 18
        scale by 3.20
    }

    set time_ns [expr {($chunk - 1) + ($source_frame + 1) * 0.002}]
    puts $manifest [format "%d,%d,%d,%.3f" $movie_frame $chunk $source_frame $time_ns]
    set start_ms [expr {$movie_frame * 100}]
    set end_ms [expr {($movie_frame + 1) * 100}]
    puts $subtitles [expr {$movie_frame + 1}]
    puts $subtitles "[srt_time $start_ms] --> [srt_time $end_ms]"
    puts $subtitles [format "Replica %s  |  %.3f ns" $replica_id $time_ns]
    puts $subtitles ""

    set outfile [file join $frame_dir [format "frame_%04d.tga" $movie_frame]]
    render TachyonInternal $outfile "-res $width $height"
    if {![file isfile $outfile]} {
        puts stderr "FATAL: render did not create $outfile"
        exit 2
    }
    puts [format "RENDERED replica=%s frame=%d time_ns=%.3f" $replica_id $movie_frame $time_ns]
}

close $manifest
close $subtitles
$reference delete
puts "COMPLETE: rendered $count frames of replica $replica_id"
quit
