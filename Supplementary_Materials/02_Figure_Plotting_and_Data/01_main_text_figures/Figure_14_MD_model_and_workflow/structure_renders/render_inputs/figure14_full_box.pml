python
from pymol import cmd
from pymol.cgo import BEGIN, LINES, TRIANGLES, ALPHA, LINEWIDTH, COLOR, VERTEX, END
cmd.reinitialize()
cmd.bg_color('white')
cmd.set('orthoscopic', 1)
cmd.set('ray_opaque_background', 0)
cmd.set('antialias', 2)
cmd.set('ray_shadows', 1)
cmd.set('connect_mode', 0)
cmd.set('stick_quality', 24)
cmd.set('sphere_quality', 2)
cmd.set('depth_cue', 0)
cmd.set('spec_reflect', .25)
cmd.set('ambient', .35)
cmd.set('direct', .65)
cmd.load('/mnt/d/work/rz_orca_pfasremoval/00-0b-Supplementary_Materials_staging/Supplementary_Materials/02_Figure_Plotting_and_Data/01_main_text_figures/Figure_14_MD_model_and_workflow/structure_renders/render_inputs/polymer.pdb', 'polymer_obj')
cmd.load('/mnt/d/work/rz_orca_pfasremoval/00-0b-Supplementary_Materials_staging/Supplementary_Materials/02_Figure_Plotting_and_Data/01_main_text_figures/Figure_14_MD_model_and_workflow/structure_renders/render_inputs/pfoa.pdb', 'pfoa')
cmd.load('/mnt/d/work/rz_orca_pfasremoval/00-0b-Supplementary_Materials_staging/Supplementary_Materials/02_Figure_Plotting_and_Data/01_main_text_figures/Figure_14_MD_model_and_workflow/structure_renders/render_inputs/sodium.pdb', 'sodium')
cmd.load('/mnt/d/work/rz_orca_pfasremoval/00-0b-Supplementary_Materials_staging/Supplementary_Materials/02_Figure_Plotting_and_Data/01_main_text_figures/Figure_14_MD_model_and_workflow/structure_renders/render_inputs/chloride.pdb', 'chloride')
if len(cmd.get_model('polymer_obj').bond) < 385:
    raise RuntimeError('Polymer connectivity incomplete; inspect topology/CONECT records')
if len(cmd.get_model('pfoa').bond) < 24:
    raise RuntimeError('PFOA connectivity incomplete; inspect topology/CONECT records')

cmd.hide('everything')
cmd.remove('(polymer_obj or pfoa) and hydro')
for element, color in {'C': '#4d4d4d', 'N': '#354f9c', 'O': '#d62728', 'F': '#47b8a6', 'H': '#f2f2f2', 'S': '#e0b33f'}.items():
    cmd.set_color('atlas_' + element, [int(color[i:i+2],16)/255 for i in (1,3,5)])
    cmd.color('atlas_' + element, 'pfoa and elem ' + element)
for element in {'C': '#4d4d4d', 'N': '#354f9c', 'O': '#d62728', 'F': '#47b8a6', 'H': '#f2f2f2', 'S': '#e0b33f'}.keys():
    cmd.color('atlas_' + element, 'polymer_obj and elem ' + element)
cmd.set('sphere_scale', .25, 'polymer_obj')
cmd.set('stick_radius', .13, 'polymer_obj')
cmd.set('stick_transparency', .10, 'polymer_obj')
cmd.set('sphere_scale', .25, 'pfoa')
cmd.set('stick_radius', .13, 'pfoa')
cmd.show('sticks', 'pfoa')
cmd.show('spheres', 'pfoa')
cmd.show('spheres', 'sodium or chloride')
cmd.set('sphere_scale', .34, 'sodium or chloride')
cmd.set_color('ion_na', [.84,.56,.22])
cmd.set_color('ion_cl', [.39,.66,.33])
cmd.color('ion_na', 'sodium')
cmd.color('ion_cl', 'chloride')
dims = [51.215, 55.693, 57.005]
center = [24.718, 27.373, 27.764]
corners = [(center[0]+x*dims[0]/2,center[1]+y*dims[1]/2,center[2]+z*dims[2]/2) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
edges = []
for i,a in enumerate(corners):
    for b in corners[i+1:]:
        if sum(a[k] != b[k] for k in range(3)) == 1:
            edges.extend([VERTEX, *a, VERTEX, *b])
cmd.load_cgo([LINEWIDTH, 1.0, BEGIN, LINES, COLOR, .34,.52,.57, *edges, END], 'unit_cell')
cmd.show('sticks', 'polymer_obj')
cmd.show('spheres', 'polymer_obj')
cmd.set('sphere_scale', .16, 'polymer_obj')
cmd.set('stick_radius', .16, 'polymer_obj')
cmd.set('stick_transparency', 0.0, 'polymer_obj')
cmd.reset()
cmd.turn('y', 8)
cmd.turn('x', -5)
cmd.zoom('unit_cell', 6.0)
view = cmd.get_view()
cmd.png('/mnt/d/work/rz_orca_pfasremoval/00-0b-Supplementary_Materials_staging/Supplementary_Materials/02_Figure_Plotting_and_Data/01_main_text_figures/Figure_14_MD_model_and_workflow/structure_renders/render_inputs/figure14_full_box_foreground.png', width=1800, height=1800, dpi=600, ray=1)
cmd.hide('everything')
faces = []
for axis in range(3):
    other = [k for k in range(3) if k != axis]
    for side in (-1, 1):
        def point(u, v):
            p = list(center)
            p[axis] += side * dims[axis] / 2
            p[other[0]] += u * dims[other[0]] / 2
            p[other[1]] += v * dims[other[1]] / 2
            return p
        quad = [point(-1,-1), point(1,-1), point(1,1), point(-1,1)]
        for a,b,c in ((quad[0],quad[1],quad[2]),(quad[0],quad[2],quad[3])):
            faces.extend([VERTEX,*a,VERTEX,*b,VERTEX,*c])
cmd.load_cgo([BEGIN, TRIANGLES, COLOR, .75,.89,.93, ALPHA, .045, *faces, END], 'cell_faces')
cmd.set_view(view)
cmd.png('/mnt/d/work/rz_orca_pfasremoval/00-0b-Supplementary_Materials_staging/Supplementary_Materials/02_Figure_Plotting_and_Data/01_main_text_figures/Figure_14_MD_model_and_workflow/structure_renders/render_inputs/figure14_cell_backdrop.png', width=1800, height=1800, dpi=600, ray=1)
python end
