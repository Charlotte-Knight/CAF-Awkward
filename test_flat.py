import uproot
import os
import awkward as ak
import vector
vector.register_awkward()
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from tqdm import tqdm

NDLArXLo = -346.9
NDLArXHi = 346.9
NDLArYLo = -215.5
NDLArYHi = 81.7
NDLArZLo = 418.2
NDLArZHi = 913.3

def read_arrays(files, prefix, fields):
  names = ["%s.%s" % (prefix, field) for field in fields]
  arrays_collection = []
  for f in tqdm(files):
    uproot_file = uproot.open(f)
    arrays_collection.append(uproot_file["cafTree"].arrays(names, library="ak"))
  arrays = ak.concatenate(arrays_collection, axis=0)

  arrays = ak.zip(
      {field.replace(".", "_"): arrays["%s.%s" % (prefix, field)] for field in fields} 
    )
  return arrays

files_dir = "files_flat/"
files = [os.path.join(files_dir, f) for f in os.listdir(files_dir)][:]

part_fields = ["pdg", "start.x", "start.y", "start.z", "end.x", "end.y", "end.z", "contained", "primary"]

vtx = read_arrays(files, "rec.common.ixn.dlp.vtx", ["x", "y", "z"])
npart = read_arrays(files, "rec.common.ixn.dlp.part", ["ndlp"]).ndlp
part = read_arrays(files, "rec.common.ixn.dlp.part.dlp", part_fields)

# restructure the arrays so it is like part[ispill][inxi][inpart]
nspill = ak.num(vtx)
part = ak.flatten(part)
npart = ak.flatten(npart)
part = ak.unflatten(part, npart)
part = ak.unflatten(part, nspill)

# flatten so that it is like part[ievent][inpart]
# this is easier for the selection that is being done here
vtx = ak.flatten(vtx)
part = ak.flatten(part)

start = ak.zip(
  {
    "x": part.start_x,
    "y": part.start_y,
    "z": part.start_z
  },
  with_name="Vector3D",
)

end = ak.zip(
  {
    "x": part.end_x,
    "y": part.end_y,
    "z": part.end_z
  },
  with_name="Vector3D",
)

part = ak.zip(
    {
      "pdg": part.pdg,
      "start": start,
      "end": end,
      "contained": part.contained,
      "primary": part.primary
    },
)

# Apply selections

o = 25 # offset
isFV = (vtx.x > NDLArXLo + o) & (vtx.x < NDLArXHi - o) & \
       (vtx.y > NDLArYLo + o) & (vtx.y < NDLArYHi - o) & \
       (vtx.z > NDLArZLo + o) & (vtx.z < NDLArZHi - o)

primary_contained = ak.all(ak.where(part.primary, part.contained, True), axis=1)

track_length = (part.end - part.start).mag
track_length = ak.nan_to_num(track_length, posinf=0) # set inf lengths to 0

longest_idx = ak.argmax(track_length, axis=1, keepdims=True)
longest_is_muon = part.pdg[longest_idx][:,0] == 13

selection = isFV & primary_contained & longest_is_muon

all_track_lengths = ak.flatten(track_length[selection])
all_track_lengths = all_track_lengths[all_track_lengths > 0]

max_track_lengths = ak.max(track_length[selection], axis=1)
max_track_lengths = max_track_lengths[max_track_lengths > 0]

plt.hist(all_track_lengths, bins=50, range=(0, 500))
plt.xlabel("Track Length (cm)")
plt.ylabel("Counts")
plt.savefig("track_length_flat.png")
plt.clf()

plt.hist(max_track_lengths, bins=50, range=(0, 500))
plt.xlabel("Max Track Length (cm)")
plt.ylabel("Counts")
plt.savefig("max_track_length_flat.png")



