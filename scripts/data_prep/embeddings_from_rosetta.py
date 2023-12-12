from Bio import SeqIO
import pathlib
import tqdm
import warnings
import esm
import torch
from numpy.lib.format import open_memmap
import numpy as np
import zarr
import dask
import dask.distributed
from dask import array as dda

warnings.filterwarnings('ignore')

def get_embeddings_from_pdb_file(model, batch_converter, pdb_file, indx_start, indx_end):
  protein_dict = {}

  with open(pdb_file, 'r') as pdb_file:
      for record in SeqIO.parse(pdb_file, 'pdb-atom'):
          protein_dict['protein 1'] = str(record.seq)

  data = [(k, v) for k, v in protein_dict.items()]
  # Load ESM-2 model

  batch_labels, batch_strs, batch_tokens = batch_converter(data)
  batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

  # Extract per-residue representations (on CPU)
  with torch.inference_mode():
      results = model(batch_tokens.cuda(), repr_layers=list(range(33)), return_contacts=False)
      token_representations = [results["representations"][i][0] for i in range(33)]

  embedding = torch.cat(token_representations)

  return embedding, indx_end, indx_end + len(embedding)

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.eval()  # disables dropout for deterministic results
model = model.cuda()

indx_start, indx_end = 0, 0

pdb_directory = pathlib.Path('/home/ajl/work/d2/code/plm/trrosetta/pdb')
all_pdb = sorted(pdb_directory.glob('*pdb'))

#create mmap of specified size
mmap = open_memmap('/home/ajl/work/d1/plm/rosetta.npy', mode='w+', dtype=float, shape=(3908474*33 + len(all_pdb)*2*33, 1280)) # generated this number from a jnotebook
offsets = []

for pdb_file in tqdm.tqdm(all_pdb):
    x, indx_start, indx_end = get_embeddings_from_pdb_file(model, batch_converter, pdb_file, indx_start, indx_end) # some arbitary (L x 33) x 1280 (embedding dim) matrix
    protlen = (len(x) // 33) 
    
    mmap[indx_start:indx_end, :] = x.cpu().numpy()
    offsets.append(protlen)
    
# to save as dask;
    
# def load_npy_chunk(da, fp, block_info=None, mmap_mode='r'):
#     """Load a slice of the .npy array, making use of the block_info kwarg"""
#     np_mmap = np.load(fp, mmap_mode=mmap_mode)
#     array_location = block_info[0]['array-location']
#     dim_slicer = tuple(list(map(lambda x: slice(*x), array_location)))
#     return np_mmap[dim_slicer]

# def dask_read_npy(fp, chunks=None, mmap_mode='r'):
#     """Read metadata by opening the mmap, then send the read job to workers"""
#     np_mmap = np.load(fp, mmap_mode=mmap_mode)
#     da = dda.empty_like(np_mmap, chunks=chunks)
#     return da.map_blocks(load_npy_chunk, fp=fp, mmap_mode=mmap_mode, meta=da)

# mmap = dask_read_npy('/home/ajl/work/d1/plm/rosetta.npy', chunks=(1000, 1280), mmap_mode='r')
# mmap.to_zarr('/home/ajl/work/d1/plm/rosettazr.zarr', mode='w')