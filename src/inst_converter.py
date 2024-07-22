import os
import re
import argparse

# args parser
parser = argparse.ArgumentParser(description='Convert .dat files into .dzn files')
parser.add_argument("-ddat", "--dat_dir", type=str, default="../Instances/",
                    help='the relative/absolute path to the directory containing .dat files to convert.')
parser.add_argument("-ddzn", "--dzn_dir", type=str, default="../Instances_dzn/",
                    help='the relative/absolute path to the directory containing .dzn files to convert.')
args = parser.parse_args()

# Hypers
dat_dir = args.dat_dir
dzn_dir = args.dzn_dir
dat_prefix = "inst"
dat_suffix = "dat"

# array stringyfier
def str2array(arr, include_brkts=True):
    str_arr = [int(i) for i in re.split('\s+', arr.strip())] # split by spaces (even on multiple occureces)
    return str_arr if include_brkts else str(str_arr)[1:-1]

# dat to dzn function
def dat2dzn(dat_dir, dat_filename, dzn_dir, suffix="dat"):
    # read dat file
    with open(dat_dir+dat_filename,"r") as dat_file:
        dat = dat_file.read().split("\n")
    # write dzn file
    dzn_filename = dat_filename.replace(suffix, "dzn")
    print(f"Writing {dzn_dir+dzn_filename}")
    with open(dzn_dir+dzn_filename,"w") as dzn_file:
        dzn_file.write(f"m = {dat[0]};")
        dzn_file.write(f"\n\n")
        dzn_file.write(f"n = {dat[1]};")
        dzn_file.write(f"\n\n")
        dzn_file.write(f"l = {str2array(dat[2])};")
        dzn_file.write(f"\n\n")
        dzn_file.write(f"s = {str2array(dat[3])};")
        dzn_file.write(f"\n\n")
        dzn_file.write(f"D = [")
        for r in range(int(dat[1])+1):
            dzn_file.write(f"|{str2array(dat[3+r+1], include_brkts=False)}\n")
        dzn_file.write(f"|];")

# loop on files
for f in os.listdir(dat_dir):
    # convert only dat files
    if f.endswith(dat_suffix):
        print(f"Converting {dat_dir+f}")
        dat2dzn(dat_dir, f, dzn_dir)

