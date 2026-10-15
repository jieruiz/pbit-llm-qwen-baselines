"""Executable state/threshold/replay contract, not a numeric LLM surrogate."""
import hashlib
import json


class SharedLfsr32:
    # Port zero of the existing rng32x64 manifest, before 16-step jump update.
    TAPS=(5,27,19,26,8,30,18,31)
    def __init__(self,material):
        if not isinstance(material,bytes) or len(material)<32:raise ValueError('32-byte root material')
        self.material=material;self.state=1;self.saved=None;self.identity=None
        self.updates=0

    def begin_tile(self,identity,rekey=True):
        if self.saved is not None:raise RuntimeError('unfinished tile')
        if rekey:
            tag=json.dumps(['input-share64-v2',identity],separators=(',',':')).encode()
            self.state=int.from_bytes(hashlib.shake_256(self.material+tag).digest(4),'little') or 1
        self.saved=self.state;self.identity=identity;self.probabilities=None;self.samples=None

    def wave(self,probabilities,samples):
        if self.saved is None:raise RuntimeError('no tile')
        q=tuple(probabilities)
        if not isinstance(samples,int) or samples<1:raise ValueError('positive sample count')
        if self.samples is not None and samples!=self.samples:raise ValueError('frame length changed during replay')
        self.samples=samples
        if len(q)!=64 or any(not isinstance(v,int) or not 0<=v<256 for v in q):raise ValueError('64 A8 codes')
        if self.probabilities is not None and q!=self.probabilities:raise ValueError('threshold changed during replay')
        self.probabilities=q;state=self.saved;counts=[0]*64;words=[]
        for _ in range(samples):
            word=sum(((state>>b)&1)<<i for i,b in enumerate(self.TAPS))
            words.append(word)
            for i,v in enumerate(q):counts[i]+=int(word<v)
            for _ in range(16):state=(state>>1)^(0x80200003 if state&1 else 0)
        self.state=state;self.updates+=samples
        return counts,words

    def end_tile(self):
        if self.saved is None:raise RuntimeError('no tile')
        self.saved=None;self.identity=None
