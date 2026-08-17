import json,struct,subprocess,sys
url=sys.argv[1]; names=sys.argv[2].split(',')
n=subprocess.run(['curl','-sSL','--max-time','60','-r','0-7',url],capture_output=True).stdout
ln=struct.unpack('<Q',n[:8])[0]
j=subprocess.run(['curl','-sSL','--max-time','180','-r',f'8-{8+ln-1}',url],capture_output=True).stdout
h=json.loads(j.decode()); base=8+ln
def bf16(b):
    out=[]
    for i in range(0,len(b),2):
        u=b[i]|(b[i+1]<<8)
        out.append(struct.unpack('<f',struct.pack('<I',u<<16))[0])
    return out
for nm in names:
    if nm not in h: print(nm,'MISSING'); continue
    v=h[nm]; s,e=v['data_offsets']
    cnt=min(16, (e-s)//2)
    raw=subprocess.run(['curl','-sSL','--max-time','120','-r',f'{base+s}-{base+s+cnt*2-1}',url],capture_output=True).stdout
    print(nm, v['dtype'], v['shape'], [round(x,6) for x in bf16(raw)])
