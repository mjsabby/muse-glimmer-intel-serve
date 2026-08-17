import struct, sys, subprocess
FM={0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
url=sys.argv[1]; want=sys.argv[2]; nbytes=int(sys.argv[3])
buf=subprocess.run(['curl','-sSL','--max-time','300','-r',f'0-{nbytes-1}',url],capture_output=True).stdout
o=0
def rd(fmt):
    global o
    n=struct.calcsize(fmt); v=struct.unpack_from(fmt,buf,o); o+=n; return v
rd('<4sI'); n_tensors,n_kv=rd('<QQ')
def rstr():
    global o
    (l,)=rd('<Q'); s=buf[o:o+l].decode('utf-8','replace'); o+=l; return s
def rval(t):
    global o
    if t==8: return rstr()
    if t==9:
        (et,)=rd('<I'); (n,)=rd('<Q')
        if et==8: return [rstr() for _ in range(n)]
        return [rd(FM[et])[0] for _ in range(n)]
    return rd(FM[t])[0]
kv={}
for i in range(n_kv):
    k=rstr(); (t,)=rd('<I'); kv[k]=rval(t)
align=kv.get('general.alignment',32)
tinfo={}
for i in range(n_tensors):
    name=rstr(); (nd,)=rd('<I'); dims=[rd("<Q")[0] for _ in range(nd)]; (tt,)=rd('<I'); (off,)=rd('<Q')
    tinfo[name]=(dims,tt,off)
data_start=(o+align-1)//align*align
print('data_start',data_start,'align',align)
for name in want.split(','):
    if name not in tinfo: print(name,'MISSING'); continue
    dims,tt,off=tinfo[name]
    n=1
    for d in dims: n*=d
    if tt!=0: print(name,'dims',dims,'ggmltype',tt,'(not F32, skipping value read)'); continue
    abs_off=data_start+off
    raw=subprocess.run(['curl','-sSL','--max-time','120','-r',f'{abs_off}-{abs_off+n*4-1}',url],capture_output=True).stdout
    vals=struct.unpack(f'<{n}f',raw[:n*4])
    import statistics
    print(f'{name} dims={dims} n={n} min={min(vals):.9g} max={max(vals):.9g} mean={statistics.fmean(vals):.9g} first8={[round(v,6) for v in vals[:8]]}')
