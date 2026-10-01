"""Panel letter alignment"""

def check_letters(fig, axes, columns=(), rows=(), tolerance_pt=.25):
    fig.canvas.draw()
    render=fig.canvas.get_renderer()
    boxes=[]
    horizontal=[]
    for ax in axes:
        hits=[text for text in ax.texts if text.get_text() in 'ABCDEF' and len(text.get_text())==1]
        if len(hits)!=1:
            raise ValueError(f'Expected exactly one panel letter in {ax}')
        boxes.append(hits[0].get_window_extent(render))
        horizontal.append('x1' if hits[0].get_ha()=='right' else 'x0')
    for pairs, coordinate in ((columns,None),(rows,'y0')):
        for i,j in pairs:
            edge = coordinate or horizontal[i]
            if coordinate is None and horizontal[i]!=horizontal[j]:
                raise ValueError('Mixed left/right panel letter anchors')
            gap=abs(getattr(boxes[i],edge)-getattr(boxes[j],edge))*72/fig.dpi
            if gap>tolerance_pt:
                raise ValueError(f'Panel letters {i+1}/{j+1} {edge} offset: {gap:.2f} pt')
    return boxes
