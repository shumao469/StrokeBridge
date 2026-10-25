"""Group Artifact Tool native objects by scientific panel; preserve editable children."""
from pathlib import Path
import argparse, json, zipfile, xml.etree.ElementTree as ET

P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
for prefix,uri in [('p',P),('a',A),('r',R)]:ET.register_namespace(prefix,uri)
ns={'p':P,'a':A}
def tag(uri,name):return '{'+uri+'}'+name

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);ap.add_argument('--receipt',required=True)
    args=ap.parse_args();target=Path(args.output);assert not target.exists()
    receipts=[]
    with zipfile.ZipFile(args.input) as source,zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as out:
        for member in source.infolist():
            data=source.read(member.filename)
            if member.filename.startswith('ppt/slides/slide') and member.filename.endswith('.xml'):
                root=ET.fromstring(data);tree=root.find('.//p:spTree',ns);assert tree is not None
                assert not root.findall('.//p:pic',ns),'Raster/vector picture is not native editable evidence'
                members={};order=[];next_id=max(int(x.attrib['id']) for x in root.findall('.//p:cNvPr',ns))+1
                for child in list(tree)[2:]:
                    name=child.find('.//p:cNvPr',ns)
                    key=name.attrib.get('name','').split('__')[0] if name is not None else ''
                    if key.startswith('panel_'):
                        if key not in members:members[key]=[];order.append(key)
                        members[key].append(child)
                assert len(members)>=6,(member.filename,len(members))
                groups={}
                for key,items in members.items():
                    xs=[];ys=[];xx=[];yy=[]
                    for item in items:
                        xf=item.find('./p:spPr/a:xfrm',ns);assert xf is not None
                        off=xf.find('a:off',ns);ext=xf.find('a:ext',ns)
                        x,y=int(off.attrib['x']),int(off.attrib['y']);w,h=int(ext.attrib['cx']),int(ext.attrib['cy'])
                        xs.append(x);ys.append(y);xx.append(x+w);yy.append(y+h)
                    x,y=min(xs),min(ys);w,h=max(xx)-x,max(yy)-y
                    group=ET.Element(tag(P,'grpSp'));nv=ET.SubElement(group,tag(P,'nvGrpSpPr'))
                    ET.SubElement(nv,tag(P,'cNvPr'),id=str(next_id),name='Panel '+key[6:]);next_id+=1
                    ET.SubElement(nv,tag(P,'cNvGrpSpPr'));ET.SubElement(nv,tag(P,'nvPr'))
                    gp=ET.SubElement(group,tag(P,'grpSpPr'));xf=ET.SubElement(gp,tag(A,'xfrm'))
                    for name,attrs in [('off',{'x':x,'y':y}),('ext',{'cx':w,'cy':h}),('chOff',{'x':x,'y':y}),('chExt',{'cx':w,'cy':h})]:
                        ET.SubElement(xf,tag(A,name),{k:str(v) for k,v in attrs.items()})
                    for item in items:group.append(item)
                    groups[key]=group
                inserted=set()
                for child in list(tree)[2:]:
                    name=child.find('.//p:cNvPr',ns);key=name.attrib.get('name','').split('__')[0] if name is not None else ''
                    if key in members:
                        ix=list(tree).index(child);tree.remove(child)
                        if key not in inserted:tree.insert(ix,groups[key]);inserted.add(key)
                receipts.append({'slide':member.filename,'panels':len(groups),'native_texts':len(root.findall('.//a:t',ns)),
                    'native_paths':len(root.findall('.//a:custGeom',ns)),'pictures':0,'group_names':['Panel '+s[6:] for s in order]})
                data=ET.tostring(root,encoding='utf-8',xml_declaration=True)
            out.writestr(member,data)
    assert receipts and all(r['native_texts']>10 and r['native_paths']>=6 for r in receipts)
    Path(args.receipt).write_text(json.dumps({'state':'native_panel_grouping_PASS','slides':receipts},indent=2),encoding='utf-8')
    print('NATIVE_PANEL_GROUPING_PASS',len(receipts))
if __name__=='__main__':main()
