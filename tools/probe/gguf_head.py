import struct, sys, subprocess
GT={0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
SZ={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
FM={0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
url=sys.argv[1]; nbytes=int(sys.argv[2]) if len(sys.argv)>2 else 12_000_000
buf=subprocess.run(['curl','-sSL','--max-time','300','-r',f'0-{nbytes-1}',url],capture_output=True).stdout
o=0
def rd(fmt):
    global o
    n=struct.calcsize(fmt); v=struct.unpack_from(fmt,buf,o); o+=n; return v
magic,ver=rd('<4sI')
n_tensors,n_kv=rd('<QQ')
print('magic',magic,'ver',ver,'tensors',n_tensors,'kv',n_kv)
def rstr():
    global o
    (l,)=rd('<Q'); s=buf[o:o+l].decode('utf-8','replace'); o+=l; return s
def rval(t):
    global o
    if t==8: return rstr()
    if t==9:
        (et,)=rd('<I'); (n,)=rd('<Q')
        if et==8: return [rstr() for _ in range(n)]
        out=[]
        for _ in range(n): out.append(rd(FM[et])[0])
        return out
    return rd(FM[t])[0]
kv={}
for i in range(n_kv):
    k=rstr(); (t,)=rd('<I'); v=rval(t); kv[k]=v
for k,v in kv.items():
    if isinstance(v,list):
        print(f'{k}: [len {len(v)}] {v[:8]}{" ..." if len(v)>8 else ""}')
    else:
        s=str(v)
        print(f'{k}: {s[:400]}{"..." if len(s)>400 else ""}')
print('--- first 40 tensors ---')
import re
seen={}
for i in range(n_tensors):
    name=rstr(); (nd,)=rd('<I'); dims=[rd("<Q")[0] for _ in range(nd)]; (tt,)=rd('<I'); (off,)=rd('<Q')
    p=re.sub(r'\.\d+\.','.{N}.',name)
    if p not in seen: seen[p]=(dims,tt)
for p,(d,t) in sorted(seen.items()): print(p,d,'ggmltype',t)
