"""Figure 14 PyMOL frame views"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import subprocess
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'input_data/replica01_prod10_frame428.pdb'
TOPOLOGY = HERE / 'input_data/prepared_associated_full_system.prmtop'
WORK = HERE / 'structure_renders/render_inputs'
OUT = HERE / 'structure_renders'
PALETTE = {'C':'#4d4d4d', 'N':'#354f9c', 'O':'#d62728',
           'F':'#47b8a6', 'H':'#f2f2f2', 'S':'#e0b33f'}
CAMERA_B_VIEW = (
    .0016280642, .3488400578, -.9371808767,
    -.9964369535, .0795942843, .0278958008,
    .0843254104, .9337962270, .3477267027,
    0.0, 0.0, -63.4735260,
    24.5099048615, 35.6587181091, 20.0016632080,
    50.0430107117, 76.9040374756, 20.0,
)
CAMERA_B_HALF_WIDTH = 12.0
CAMERA_B_HALF_DEPTH = 10.0
WATER_SAMPLE_FRACTION = .30
WATER_SAMPLE_SEED = 'Figure14B-v1'

def xyz(line):
    return np.array([float(line[i:i+8]) for i in (30, 38, 46)])

def heavy(line):
    atom = line[12:16].strip()
    return not atom.startswith('H')

def near(point, sites, cutoff):
    return bool(np.any(np.sum((sites-point)**2, axis=1) <= cutoff**2))

def topology_bonds():
    blocks=[]; active=False; values=[]
    for line in TOPOLOGY.read_text().splitlines():
        if line.startswith('%FLAG'):
            if active:blocks.append(values)
            active=line.split()[1] in ('BONDS_INC_HYDROGEN','BONDS_WITHOUT_HYDROGEN')
            values=[]
        elif active and not line.startswith('%FORMAT'):
            values.extend(map(int,line.split()))
    if active:blocks.append(values)
    bonds=set()
    for values in blocks:
        for i in range(0,len(values),3):
            a,b=values[i]//3+1,values[i+1]//3+1
            if a!=b:bonds.add(tuple(sorted((a,b))))
    if not bonds:raise ValueError('Amber topology contains no bonds')
    return bonds

def write_bonded_subset(name, selection, bonds):
    target=WORK/(name+'.pdb')
    serials={int(line[6:11]):i for i,line in enumerate(selection,1)}
    if len(serials)!=len(selection):raise ValueError(f'Duplicate PDB serials in {name}')
    out=[line[:6]+f'{i:5d}'+line[11:] for i,line in enumerate(selection,1)]
    neighbors=defaultdict(set)
    for ai,aj in bonds:
        if ai in serials and aj in serials:
            i,j=serials[ai],serials[aj]
            neighbors[i].add(j);neighbors[j].add(i)
    out.append('TER')
    for i in sorted(neighbors):
        partners=sorted(neighbors[i])
        for k in range(0,len(partners),4):
            out.append('CONECT'+f'{i:5d}'+''.join(f'{j:5d}' for j in partners[k:k+4]))
    out.append('END')
    target.write_text('\n'.join(out)+'\n')

def write_pml(name, local, cell, center):
    dims = [float(cell[i:i+9]) for i in (6,15,24)]
    top = f'''from pymol import cmd
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
'''
    objects = ('polymer','pfoa','water','sodium','chloride') if local else ('polymer','pfoa','sodium','chloride')
    for obj in objects:
        path=WORK/(obj+('_local' if local and obj in ('water','sodium','chloride') else '')+'.pdb')
        object_name='polymer_obj' if obj == 'polymer' else obj
        if path.read_text().startswith('TER\nEND'):
            top+=f"cmd.select({object_name!r}, 'none')\n"
        else:
            top+=f"cmd.load({str(path.resolve())!r}, {object_name!r})\n"
    top += '''if len(cmd.get_model('polymer_obj').bond) < 385:
    raise RuntimeError('Polymer connectivity incomplete; inspect topology/CONECT records')
if len(cmd.get_model('pfoa').bond) < 24:
    raise RuntimeError('PFOA connectivity incomplete; inspect topology/CONECT records')
''' + ('''cmd.unbond('water and elem H', 'water and elem H')
cmd.unbond('water and elem O', 'water and elem O')
if len(cmd.get_model('water').bond) != 2 * cmd.count_atoms('water') // 3:
    raise RuntimeError('Water connectivity differs from two O-H bonds per molecule')
''' if local else '') + '''
cmd.hide('everything')
cmd.remove('(polymer_obj or pfoa) and hydro')
for element, color in ''' + repr(PALETTE) + '''.items():
    cmd.set_color('atlas_' + element, [int(color[i:i+2],16)/255 for i in (1,3,5)])
    cmd.color('atlas_' + element, 'pfoa and elem ' + element)
for element in ''' + repr(PALETTE) + '''.keys():
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
'''
    if not local:
        top += f'''dims = {dims!r}
center = {list(center)!r}
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
cmd.png(''' + repr(str((WORK / 'figure14_full_box_foreground.png').resolve())) + ''', width=1800, height=1800, dpi=600, ray=1)
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
cmd.png(''' + repr(str((WORK / 'figure14_cell_backdrop.png').resolve())) + ''', width=1800, height=1800, dpi=600, ray=1)
'''
    else:
        top += '''cmd.show('sticks', 'polymer_obj')
cmd.show('spheres', 'polymer_obj')
cmd.show('sticks', 'water')
cmd.show('spheres', 'water')
cmd.set('sphere_scale', .075, 'water')
cmd.set('sphere_scale', .05, 'water and elem H')
cmd.set('stick_radius', .055, 'water')
cmd.set('stick_transparency', .34, 'water')
cmd.set('sphere_transparency', .40, 'water')
cmd.color('atlas_O', 'water and elem O')
cmd.color('atlas_H', 'water and elem H')
cmd.set_view(''' + repr(CAMERA_B_VIEW) + ''')
'''
    if local:
        top += f"cmd.png({str((OUT / name).resolve())!r}, width=1800, height=1800, dpi=600, ray=1)\n"
    pml = WORK / name.replace('.png','.pml')
    pml.write_text('python\n'+top+'python end\n')
    return pml

def main(prepare_only=False, mamba=Path('/home/zwa/miniforge3/bin/mamba'), env='pymol-render'):
    lines = SOURCE.read_text().splitlines()
    cell = next(line for line in lines if line.startswith('CRYST1'))
    atoms = [line for line in lines if line.startswith(('ATOM  ', 'HETATM'))]
    groups = defaultdict(list)
    for line in atoms:
        groups[line[17:20].strip()].append(line)
    if len(groups['PVB']) != 374 or len(groups['PFO']) != 25:
        raise ValueError('Expected one validated 12-unit polymer and one PFOA')
    box = np.array([float(cell[i:i+9]) for i in (6,15,24)])
    solute = np.array([xyz(line) for line in groups['PVB']+groups['PFO']])
    center = (solute.min(axis=0)+solute.max(axis=0))/2
    def reimage(line,shift=None):
        point=xyz(line)
        if shift is None:shift=np.rint((point-center)/box)*box
        position=point-shift
        return line[:30]+''.join(f'{v:8.3f}' for v in position)+line[54:]
    whole_waters=defaultdict(list)
    for line in groups['WAT']:whole_waters[line[21:27]].append(line)
    groups['WAT']=[]
    for water in whole_waters.values():
        oxygen=next((line for line in water if line[12:16].strip().startswith('O')),water[0])
        shift=np.rint((xyz(oxygen)-center)/box)*box
        groups['WAT'].extend(reimage(line,shift) for line in water)
    for res in ('Na+','Cl-'):
        groups[res]=[reimage(line) for line in groups[res]]
    sites = np.array([xyz(line) for line in groups['PFO'] if heavy(line)])
    waters = defaultdict(list)
    for line in groups['WAT']:
        waters[line[21:27]].append(line)
    rotation = np.array(CAMERA_B_VIEW[:9]).reshape(3, 3)
    view_origin = np.array(CAMERA_B_VIEW[12:15])
    eligible = []
    for key, water in waters.items():
        oxygen = next(line for line in water if line[12:16].strip().startswith('O'))
        camera_xyz = rotation @ (xyz(oxygen) - view_origin)
        if (abs(camera_xyz[0]) <= CAMERA_B_HALF_WIDTH
                and abs(camera_xyz[1]) <= CAMERA_B_HALF_WIDTH
                and abs(camera_xyz[2]) <= CAMERA_B_HALF_DEPTH):
            eligible.append(key)
    def sampled(key, seed, fraction):
        digest = hashlib.sha256(f'{seed}:{key}'.encode()).digest()
        return int.from_bytes(digest[:8], 'big') / 2**64 < fraction
    selected = {key for key in eligible if sampled(key, WATER_SAMPLE_SEED, WATER_SAMPLE_FRACTION)}
    display_water = [line for key, water in waters.items() if key in selected for line in water]
    local_ions = {res:[line for line in groups[res] if near(xyz(line), sites, 8.0)]
                  for res in ('Na+','Cl-')}
    WORK.mkdir(parents=True,exist_ok=True)
    bonds=topology_bonds()
    for obsolete in ('water.pdb','water_local.pdb','water_ambient.pdb'):
        (WORK / obsolete).unlink(missing_ok=True)
    for name, selection in [('polymer',groups['PVB']),('pfoa',groups['PFO']),
                             ('water_local',display_water),('sodium',groups['Na+']),
                             ('chloride',groups['Cl-']),
                             ('sodium_local',local_ions['Na+']),('chloride_local',local_ions['Cl-'])]:
        write_bonded_subset(name,selection,bonds)
    with (WORK/'selection_audit.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(('component','total_or_eligible','drawn','selection'))
        writer.writerow(('panel A waters',len(waters),0,'not displayed in overview'))
        writer.writerow(('panel B eligible waters',len(eligible),len(selected),
                         f'oxygen in fixed camera |u|,|v| <= {CAMERA_B_HALF_WIDTH:g} Å and |depth| <= {CAMERA_B_HALF_DEPTH:g} Å; '
                         f'SHA-256 seed {WATER_SAMPLE_SEED!r} on residue ID; uniform fraction {WATER_SAMPLE_FRACTION:.2f}; all sampled waters complete'))
        for name,full,local,rule in [('Na+ ions',len(groups['Na+']),len(local_ions['Na+']),'<= 8 Å from PFOA heavy atom'),
                                     ('Cl- ions',len(groups['Cl-']),len(local_ions['Cl-']),'<= 8 Å from PFOA heavy atom')]:
            writer.writerow((name,full,local,rule))
    scripts = [write_pml('figure14_full_box.png',False,cell,center),
               write_pml('figure14_hydrated_site.png',True,cell,center)]
    print('Prepared PyMOL scripts and selection_audit.csv:',scripts)
    if prepare_only: return
    runner = mamba
    if not runner.exists():
        raise FileNotFoundError(f'PyMOL mamba runner absent: {runner}; use --prepare-only for audit')
    for script in scripts:
        subprocess.run([str(runner),'run','-n',env,'pymol','-cq',str(script)],check=True)
    with Image.open(WORK/'figure14_cell_backdrop.png') as backdrop, Image.open(WORK/'figure14_full_box_foreground.png') as foreground:
        if backdrop.size != foreground.size:
            raise ValueError('Cell and molecular camera passes have different dimensions')
        mask = backdrop.convert('RGBA').getchannel('A').point(lambda a: min(170, 12*a))
        tint = Image.new('RGBA', backdrop.size, (210, 235, 243, 0))
        tint.putalpha(mask)
        Image.alpha_composite(tint, foreground.convert('RGBA')).save(OUT/'figure14_full_box.png')
    for name in ('figure14_full_box.png','figure14_hydrated_site.png'):
        if not (OUT/name).is_file(): raise FileNotFoundError(OUT/name)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--mamba',type=Path,default=Path('/home/zwa/miniforge3/bin/mamba'))
    parser.add_argument('--env',default='pymol-render')
    args=parser.parse_args()
    main(args.prepare_only,args.mamba,args.env)
