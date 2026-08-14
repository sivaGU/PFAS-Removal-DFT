# VMD rendering

proc safe {cmd} {
    if {[catch {eval $cmd} err]} {
        puts "WARN: $cmd -> $err"
    }
}

if {$argc < 5} {
    puts "ERROR: expected topology trajectory outdir mode max_frames"
    quit
}

set topology [lindex $argv 0]
set trajectory [lindex $argv 1]
set outdir [lindex $argv 2]
set mode [lindex $argv 3]
set max_frames [lindex $argv 4]

file mkdir $outdir

mol new $topology type parm7 waitfor all
mol addfile $trajectory type dcd waitfor all

set nframes [molinfo top get numframes]
if {$nframes < 1} {
    puts "ERROR: trajectory contains no frames"
    quit
}

set stride [expr {int(ceil(double($nframes) / double($max_frames)))}]
if {$stride < 1} {
    set stride 1
}

mol delrep 0 top

safe {axes location Off}
safe {display projection Orthographic}
safe {display depthcue off}
safe {display ambientocclusion on}
safe {color Display Background 8}

if {$mode eq "pfoa_assoc_zoom"} {
    set r48_scope "resname R48 and within 10 of resname PFO"
    set cl_scope "element Cl and within 12 of resname PFO"
} else {
    set r48_scope "resname R48"
    set cl_scope "element Cl"
}

# Resin
mol representation Licorice 0.08 8 8
mol color ColorID 2
mol selection "$r48_scope and element C"
mol material Transparent
mol addrep top

mol representation Licorice 0.14 12 12
mol color ColorID 0
mol selection "$r48_scope and element N"
mol material Opaque
mol addrep top

mol representation Licorice 0.08 8 8
mol color Element
mol selection "$r48_scope and not hydrogen and not element C N"
mol material Transparent
mol addrep top

if {$mode eq "pfoa_assoc" || $mode eq "pfoa_assoc_zoom"} {
    # PFOA
    mol representation Licorice 0.25 16 16
    mol color Element
    mol selection "resname PFO and not hydrogen"
    mol material Opaque
    mol addrep top

    # Tail waters
    mol representation VDW 0.18 8
    mol color ColorID 0
    if {$mode eq "pfoa_assoc_zoom"} {
        mol selection "water and element O and within 6 of (resname PFO and (element F or name C1 C2 C3 C4 C5 C6 C7))"
    } else {
        mol selection "water and element O and within 5 of (resname PFO and (element F or name C1 C2 C3 C4 C5 C6 C7))"
    }
    mol material Transparent
    mol addrep top
}

# Chloride
mol representation VDW 0.55 16
mol color ColorID 7
mol selection "$cl_scope"
mol material Opaque
mol addrep top

set align_sel "resname R48 and not hydrogen"
set ref [atomselect top $align_sel frame 0]

set rendered 0
for {set frame 0} {$frame < $nframes} {incr frame $stride} {
    animate goto $frame

    # Alignment
    set mobile [atomselect top $align_sel frame $frame]
    set all [atomselect top all frame $frame]
    set transform [measure fit $mobile $ref]
    $all move $transform
    $mobile delete
    $all delete

    if {$mode eq "pfoa_assoc_zoom"} {
        set focus [atomselect top "(resname PFO or (resname R48 and not hydrogen and within 8 of resname PFO) or (element Cl and within 8 of resname PFO) or (water and element O and within 6 of resname PFO))" frame $frame]
        set center [measure center $focus]
        $focus delete
        set all [atomselect top all frame $frame]
        $all moveby [vecscale -1 $center]
        $all delete
    }

    display resetview
    rotate x by -70
    rotate z by 25
    if {$mode eq "pfoa_assoc_zoom"} {
        scale by 4.15
    } elseif {$mode eq "pfoa_assoc"} {
        scale by 1.45
    } else {
        scale by 1.35
    }

    set outfile [format "%s/frame_%04d.tga" $outdir $rendered]
    puts "Rendering frame $frame to $outfile"
    render TachyonInternal $outfile
    incr rendered
}

$ref delete
puts "Rendered $rendered frames from $nframes trajectory frames with stride $stride"
quit
