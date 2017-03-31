import numpy as np

protagonist_name="RIGGAN"
dialogue_line_start = '\s{10,}'
with open('/home/chandrasekar/Desktop/birdman-script.txt') as f:
	lines = []
	is_in_diagloue = false
	current_line = []
	for line in f:
		if is_in_diagloue:
			if line.:
				pass
		else:

		if line.strip() == "RIGGAN" or line.strip() == "RIGGAN (CONT'D)" or line.strip() == "RIGGAN (O.S.)" or line.strip == "RIGGAN (O.S.) (CONT'D)":
			is_in_diagloue = true
					
