import json,subprocess,sys,struct
def hdr(url):
    n=subprocess.run(['curl','-sSL','--max-time','60','-r','0-7',url],capture_output=True).stdout
    ln=struct.unpack('<Q',n[:8])[0]
    j=subprocess.run(['curl','-sSL','--max-time','120','-r',f'8-{8+ln-1}',url],capture_output=True).stdout
    return json.loads(j.decode())
url=sys.argv[1]
h=hdr(url)
for k,v in h.items():
    if k=='__metadata__': print('META',v); continue
    print(f"{k}\t{v['dtype']}\t{v['shape']}")
