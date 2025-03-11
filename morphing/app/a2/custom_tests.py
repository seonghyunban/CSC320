# CSC320 Fall 2024
# Assignment 2
# (c) Kyros Kutulakos, Towaki Takikawa, Robin Swanson, Esther Lin
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

import os
import sys
import glob

paths = []

paths.append("data/custom/smoke")
file = "data/custom/smoke/smoke.jpeg"


supersample = True
bilinear = True
reference = False


if supersample:
    ss = " --supersampling 3"
else:
    ss = " --supersampling 1"
    
if bilinear:
    bi = " --bilinear"
else:
    bi = ""

options = ss + bi


for path in paths:
    if os.path.isdir(path):
        img_name = os.path.splitext(os.path.basename(path))[0]
        dirname = os.path.basename(path)
        cmd = f"python a2_headless.py --source-image-path {file} \
                --source-line-path {path}/source.csv \
                --destination-line-path {path}/destination.csv --output-path test_results/{dirname}_vanila"
        for arg in sys.argv[1:]:
            cmd += " " + arg
        os.system(cmd)
        
        cmd = f"python a2_headless.py --source-image-path {file} \
                --source-line-path {path}/source.csv \
                --destination-line-path {path}/destination.csv --output-path test_results/{dirname}_bilinear{bi}"
        for arg in sys.argv[1:]:
            cmd += " " + arg
        os.system(cmd)
        
        cmd = f"python a2_headless.py --source-image-path {file} \
                --source-line-path {path}/source.csv \
                --destination-line-path {path}/destination.csv --output-path test_results/{dirname}_supersampling{bi}{ss}"
        for arg in sys.argv[1:]:
            cmd += " " + arg
        os.system(cmd)
        

