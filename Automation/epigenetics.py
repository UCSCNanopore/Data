# epigenetics.py
# Contact: Jacob Schreiber
#          jmschreiber91@gmail.com

'''
'''

import pandas as pd
import numpy as np
import itertools as it
import seaborn as sns
from matplotlib import pyplot as plt
import re
import sys 

from yahmm import *
from PyPore.hmm import *
from PyPore.DataTypes import *

def EpigeneticsModel( distributions, name, low=0, high=90 ):
	"""
	Create the HMM using circuit board methodologies.
	"""

	def match_model( distribution, name ):
		"""
		Build a small match model, allowing for oversegmentation where the
		distribution representing number of segments is a mixture of two
		exponentials.
		"""

		model = Model( name=name )

		match = State( distribution, name=name ) # Match without oversegmentation
		match_os = State( distribution, name="MO{}".format(name[1:]) ) # Match with oversegmentation

		model.add_state( match )
		model.add_state( match_os )

		model.add_transition( model.start, match, 0.95 )
		model.add_transition( model.start, match_os, 0.05 )

		model.add_transition( match, match, 0.10 )
		model.add_transition( match, model.end, 0.90 )

		model.add_transition( match_os, match_os, 0.80 )
		model.add_transition( match_os, model.end, 0.20 )
		return model

	def BakeModule( distribution, i=None, low=0, high=90 ):
		"""
		Create a module which represents a full slice of the PSSM. Take in
		the distribution which should be represented at that position, and
		create a board with 7 ports on either side.
		"""
		
		board = HMMBoard( n=7, name=str(i) )

		idx = str(i) if i else ""

		delete = State( None, name="D-{}".format( idx ) )
		match = match_model( distribution, name="M-{}".format( idx ) )

		insert = State( UniformDistribution( low, high ), name="I-{}".format(idx))
		match_s = State( distribution, name="MS-{}".format( idx ))
		match_e = State( distribution, name="ME-{}".format( idx ))

		board.add_model( match )
		for state in [ delete, insert, match_s, match_e ]:
			board.add_state( state )

		board.add_transition( board.s1, delete, 1.00 )
		board.add_transition( board.s2, match.start, 1.00 )
		board.add_transition( board.s3, board.e4, 1.00 )
		board.add_transition( board.s4, match.end, 1.00 )
		board.add_transition( board.e5, match_e, 1.00 )
		board.add_transition( board.s6, match_s, 1.00 )
		board.add_transition( board.e7, match.start, 0.90 )
		board.add_transition( board.e7, match_e, 0.05 )
		board.add_transition( board.e7, board.s7, 0.05 )

		board.add_transition( delete, board.e1, 0.1 )
		board.add_transition( delete, insert, 0.1 )
		board.add_transition( delete, board.e2, 0.8 )

		board.add_transition( insert, match.start, 0.10 )
		board.add_transition( insert, insert, 0.50 )
		board.add_transition( insert, board.e1, 0.05 )
		board.add_transition( insert, board.e2, 0.35 )

		board.add_transition( match.end, insert, 0.01 )
		board.add_transition( match.end, board.e1, 0.01 )
		board.add_transition( match.end, board.e2, .96 )
		board.add_transition( match.end, board.e3, .01 )
		board.add_transition( match.end, board.s7, .01 )

		board.add_transition( match_s, board.s5, 0.80 )
		board.add_transition( match_s, match_s, 0.20 )

		board.add_transition( match_e, board.e2, 0.10 )
		board.add_transition( match_e, match_e, 0.10 )
		board.add_transition( match_e, board.e6, 0.80 )
		return board

	def BakeUModule( d_a, d_b, name=None ):
		"""
		Create a module which represents the undersegmentation part of the
		profile HMM. Take in two adjacent distributions, d_a and d_b, and
		create a middle for them. This can handle both kernel densities and
		normal distributions, returning a GaussianKernelDensity if either
		a or b are GKDs, otherwise a NormalDistribution.
		"""

		board = HMMBoard( 7, name=name )

		if 'KernelDensity' in d_a.name:
			if 'KernelDensity' in d_b.name:
				a_points, b_points = d_a.parameters[0], d_b.parameters[0]
				blend_points = [ (a+b)/2 for a in a_points for b in b_points ]
				blend_bandwidth = ( d_a.parameters[1] + d_b.parameters[1] ) / 2
				blend_match = GaussianKernelDensity( blend_points, blend_bandwidth )
			else:
				mean, std = d_b.parameters[0], d_b.parameters[1]
				blend_points = [ (mean+j)/2 for j in d_a.parameters[0] ]
				blend_bandwidth = ( std+d_a.parameters[1] ) / 2
				blend_match = GaussianKernelDensity( blend_points, blend_bandwidth )

		else:
			if 'KernelDensity' in d_b.name:
				mean, std = d_a.parameters[0], d_a.parameters[1]
				blend_points = [ (mean+b)/2 for b in d_b.parameters[0] ]
				blend_bandwidth = ( std+d_b.parameters[1] ) / 2
				blend_match = GaussianKernelDensity( blend_points, blend_bandwidth )
			else:
				a_mean, a_std = d_a.parameters[0], d_a.parameters[1]
				b_mean, b_std = d_b.parameters[0], d_b.parameters[1]

				blend_mean = ( a_mean + b_mean ) / 2
				blend_std = ( a_std + b_std ) / 2
				blend_match = NormalDistribution( blend_mean, blend_std )

		blend_state = State( blend_match, name="U-{}".format(name) )
		board.add_transition( board.s1, board.e1, 1.00 )
		board.add_transition( board.s2, board.e2, 1.00 )
		board.add_transition( board.s3, board.e3, 1.00 )
		board.add_transition( board.s4, blend_state, 1.00 )
		board.add_transition( blend_state, board.e4, 1.00 )
		board.add_transition( board.e5, board.s5, 1.00 )
		board.add_transition( board.s6, board.e6, 1.00 )
		board.add_transition( board.e7, board.s7, 1.00 )
		return board

	model = Model( name )
	boards = []

	for i, distribution in enumerate( distributions ):
		if i > 0:
			last_board = boards[-1]

		if isinstance( distribution, Distribution ):
			d_board = BakeModule( distribution, ":{}".format(i+1), low=low, high=high )
			model.add_model( d_board )

			if i == 0:
				boards.append( d_board )
				continue

			if isinstance( distributions[i-1], Distribution ):
				u_board = BakeUModule( distribution, distributions[i-1], name='U:'+str(i+1) )
				boards.append( u_board )
				boards.append( d_board )
				model.add_model( u_board )

				model.add_transition( last_board.e1, u_board.s1, 1.00 )
				model.add_transition( last_board.e2, u_board.s2, 1.00 )
				model.add_transition( last_board.e3, u_board.s3, 1.00 )
				model.add_transition( last_board.e4, u_board.s4, 1.00 )
				model.add_transition( u_board.s5, last_board.e5, 1.00 )
				model.add_transition( last_board.e6, u_board.s6, 1.00 )
				model.add_transition( u_board.s7, last_board.e7, 1.00 )

				model.add_transition( u_board.e1, d_board.s1, 1.00 )
				model.add_transition( u_board.e2, d_board.s2, 1.00 )
				model.add_transition( u_board.e3, d_board.s3, 1.00 )
				model.add_transition( u_board.e4, d_board.s4, 1.00 )
				model.add_transition( d_board.s5, u_board.e5, 1.00 )
				model.add_transition( u_board.e6, d_board.s6, 1.00 )
				model.add_transition( d_board.s7, u_board.e7, 1.00 )

			elif isinstance( distributions[i-1], dict ):
				n = len( distributions[i-1].keys() )

				for last_board in boards[-2*n+1::2]:
					key = last_board.name.split()[1].split(':')[0]
					last_dist = distributions[i-1][key]

					u_board = BakeUModule( distribution, last_dist, name="U-{}:{}".format( key, i+1 ) )
					boards.append( u_board )
					model.add_model( u_board )

					model.add_transition( last_board.e1, u_board.s1, 1.00 )
					model.add_transition( last_board.e2, u_board.s2, 1.00 )
					model.add_transition( last_board.e3, u_board.s3, 1.00 )
					model.add_transition( last_board.e4, u_board.s4, 1.00 )
					model.add_transition( u_board.s5, last_board.e5, 1.00/n, pseudocount=1e6 )
					model.add_transition( last_board.e6, u_board.s6, 1.00 )
					model.add_transition( u_board.s7, last_board.e7, 1.00/n, pseudocount=1e6 )

					model.add_transition( u_board.e1, d_board.s1, 1.00 )
					model.add_transition( u_board.e2, d_board.s2, 1.00 )
					model.add_transition( u_board.e3, d_board.s3, 1.00 )
					model.add_transition( u_board.e4, d_board.s4, 1.00 )
					model.add_transition( d_board.s5, u_board.e5, 1.00 )
					model.add_transition( u_board.e6, d_board.s6, 1.00 )
					model.add_transition( d_board.s7, u_board.e7, 1.00 )

				boards.append( d_board )

		elif isinstance( distribution, dict ):
			n = len( distribution.keys() )
			for j, (key, dist) in enumerate( distribution.items() ):
				d_board = BakeModule( dist, "{}:{}".format( key, i+1 ), low=low, high=high )
				model.add_model( d_board )

				if isinstance( distributions[i-1], dict ):
					u_board = BakeUModule( dist, distributions[i-1][key], name="U-{}:{}".format( key, i+1 ) )
					model.add_model( u_board )
					boards.append( u_board )
					boards.append( d_board )
					last_board = boards[-2*n-1]

					model.add_transition( last_board.e1, u_board.s1, 1.00 )
					model.add_transition( last_board.e2, u_board.s2, 1.00 )
					model.add_transition( last_board.e3, u_board.s3, 1.00 )
					model.add_transition( last_board.e4, u_board.s4, 1.00 )
					model.add_transition( u_board.s5, last_board.e5, 1.00 )
					model.add_transition( last_board.e6, u_board.s6, 1.00 )
					model.add_transition( u_board.s7, last_board.e7, 1.00 )

					model.add_transition( u_board.e1, d_board.s1, 1.00 )
					model.add_transition( u_board.e2, d_board.s2, 1.00 )
					model.add_transition( u_board.e3, d_board.s3, 1.00 )
					model.add_transition( u_board.e4, d_board.s4, 1.00 )
					model.add_transition( d_board.s5, u_board.e5, 1.00 )
					model.add_transition( u_board.e6, d_board.s6, 1.00 )
					model.add_transition( d_board.s7, u_board.e7, 1.00 )
				else:
					u_board = BakeUModule( dist, distributions[i-1], name="U-{}:{}".format( key, i+1 ) )
					model.add_model( u_board )
					boards.append( u_board )
					boards.append( d_board )

					model.add_transition( last_board.e1, u_board.s1, 1.00/n, pseudocount=1e6 )
					model.add_transition( last_board.e2, u_board.s2, 1.00 / n, pseudocount=1e6 )
					model.add_transition( last_board.e3, u_board.s3, 1.00/n, pseudocount=1e6 )
					model.add_transition( last_board.e4, u_board.s4, 1.00/n, pseudocount=1e6 )
					model.add_transition( u_board.s5, last_board.e5, 1.00 )
					model.add_transition( last_board.e6, u_board.s6, 1.00/n, pseudocount=1e6 )
					model.add_transition( u_board.s7, last_board.e7, 1.00 )

					model.add_transition( u_board.e1, d_board.s1, 1.00 )
					model.add_transition( u_board.e2, d_board.s2, 1.00 )
					model.add_transition( u_board.e3, d_board.s3, 1.00 )
					model.add_transition( u_board.e4, d_board.s4, 1.00 )
					model.add_transition( d_board.s5, u_board.e5, 1.00 )
					model.add_transition( u_board.e6, d_board.s6, 1.00 )
					model.add_transition( d_board.s7, u_board.e7, 1.00 )

	board = boards[0]
	initial_insert = State( UniformDistribution( low, high ), name="I:0" )
	model.add_state( initial_insert )

	model.add_transition( initial_insert, initial_insert, 0.70 )
	model.add_transition( initial_insert, board.s1, 0.1 )
	model.add_transition( initial_insert, board.s2, 0.2 )

	model.add_transition( model.start, initial_insert, 0.02 )
	model.add_transition( model.start, board.s1, 0.08 )
	model.add_transition( model.start, board.s2, 0.90 )
	model.add_transition( board.s6, model.start, 1.00  )

	board = boards[-1]
	model.add_transition( board.e1, model.end, 1.00 )
	model.add_transition( board.e2, model.end, 1.00 )

	model.bake()
	return model

def build_profile():
	'''
	Build a profile HMM based on a file of hand curated data, with forks in the
	appropriate places.

	This is a cartoon of the HMM:

	   /---mC---\	    /---mC---\   /-CAT-\
	---|----C---|-------|----C---|---|--T--|-----------
	   \--hmC---/	    \--hmC---/   \--X--/
	
	'''

	data = pd.read_excel( "data.xlsx", "Sheet1" )
	hmms, dists = [], {}

	for name, frame in data.groupby( 'Label' ):
		means, stds = frame.mean(axis=0), frame.std(axis=0)
		dists[name] = [ NormalDistribution( m, s ) for m, s in zip( means, stds ) if not np.isnan( m ) ]

	labels = ['T', 'CAT', 'X' ]
	cytosines = [ 'C', 'mC', 'hmC' ]

	profile = []
	profile.extend( dists['CAT'][::-1] )

	for i in xrange( 9 ):
		profile.append( { 'C': dists['C'][8-i], 'mC': dists['mC'][8-i], 'hmC': dists['hmC'][8-i] } )

	profile.extend( dists['CAT'][::-1] )
	profile.extend( dists['CAT'][::-1] )
	profile.extend( dists['CAT'] )
	profile.extend( dists['CAT'] )

	for i in xrange( 9 ):
		profile.append( { 'C': dists['C'][i], 'mC': dists['mC'][i], 'hmC': dists['hmC'][i] } )

	profile.extend( dists['CAT'] )
	
	for i in xrange( 6 ):
		profile.append( { 'T': dists['T'][i], 'X': dists['X'][i], 'CAT': dists['CAT'][i%3] } )

	profile.extend( dists['CAT'] )
	profile.extend( dists['CAT'] )
	profile.extend( dists['CAT'] )
	profile.extend( dists['CAT'] )

	return profile

def parse_file( filename, hmm ):
	"""
	Take in a .abf file, perform event detection, segmentation on the event.
	"""

	# Load the file
	file = File( filename )

	# Perform event detection using defailt parameters on lambda_event_parser
	# which are to take events in a band of -0.5 - 90 pA longer than 1 second.
	print "\nEntering File {}".format( filename )
	print "\tParsing File..."
	file.parse( lambda_event_parser( threshold=90, rules=[ lambda event: event.duration > 0.5*file.second, lambda event: event.min > 0. ] ) )
	print "\tFile Parse Complete."

	# Set up the parser using the Bayesian parameterization.
	parser = SpeedyStatSplit( prior_segments_per_second=40., 
		sampling_freq=1.e5, cutoff_freq=2000.  )

	# Go through each event in the file
	for n, event in enumerate( file.events ):
		print "\tParsing Event {} of {}".format( n, len( file.events ) )

		# Filter the event using a first-order Bessel filter with a cutoff
		# of 2kHz. 
		event.filter( order=1, cutoff=2000 )

		# Use the parser to parse the event.
		event.parse( parser )

		# Run the viterbi algorithm using the hmm on the event to get the log
		# probability of the ML path, and the ML path
		prob, seq = event.apply_hmm( hmm, algorithm='viterbi' )
		print "\t\tHMM Probability: {}".format( prob )

		# Filter out the silent states to get just character generating states
		# along the ML path
		seq = filter( lambda x: not x[1].is_silent(), seq )


		# Go through the event, checking to see if an undersegmented area was
		# found by going through an undersegmentation character generating
		# state, prefixed by 'U'.
		i, j = 0, 0
		u_sum = sum( state.name[0] == 'U' for _, state in seq )

		print "\t\tTotal Undersegmented Regions: {}".format( u_sum )
		while i < len( event.segments ):
			state, segment = seq[j][1], event.segments[i] 

			if state.name[0] == 'U':
				# If undersegmentation is found, it means that two adjacent
				# segments which should have been split were not, so there is
				# a single split which needs to be made. Use the parser to
				# split on the single best split in the region.

				gain, x = parser.best_single_split( segment.current )
				print "\t\tSplitting Undersegmented Region, gain {}, index {}"\
					.format( gain, x )

				# Create two new segment objects based on where the best split is.
				left_split = Segment( start=segment.start, 
									  second=file.second,
									  duration=float(x)/file.second,
									  end=segment.start+float(x)/file.second, 
									  current=segment.current[:x] )
				right_split = Segment( start=segment.start+float(x)/file.second,
									   duration=(len(segment.current)-x)/file.second,
									   end=segment.end,
						               second=file.second, 
						               current=segment.current[x:] )
					
				# Add the new segments in the appropriate place in the list of
				# segments in an event, and increase the pointers appropriately
				# to handle iterating over a list where insertions are occuring.
				if i == len( event.segments ) - 1:
					# If the undersegmentation happened on the last segment,
					# remove it and simply append the new segments to the end
					event.segments = list(event.segments[:i]) + \
						[left_split, right_split]
					break

				elif i == 0:
					# If the undersegmentation happened on the first segment,
					# remove it and put the two new segments at the beginning
					event.segments = [left_split, right_split] + \
						list( event.segments[i+1:] )
					i += 2
					j += 1
				else:
					# Else it happened in the middle, in which case split the
					# list at the appropriate place and insert them in.
					event.segments = list(event.segments[:i]) + \
						[left_split, right_split] + list(event.segments[i+1:])
					i += 2
					j += 1
			else:
				# If no undersegmentation happened, continue on.
				i += 1
				j += 1

		if u_sum > 0:
			prob, seq = event.apply_hmm( hmm, algorithm='viterbi' )

			# Filter out the silent states to get just character generating states
			# along the ML path
			seq = filter( lambda x: not x[1].is_silent(), seq )

			# Go through the event, checking to see if an undersegmented area was
			# found by going through an undersegmentation character generating
			# state, prefixed by 'U'.

			u_sum = sum( state.name[0] == 'U' for _, state in seq )
			print "\t\t{} Undersegmented Regions Left".format( u_sum )

	return file

def get_events( filenames, hmm, force_parse=False ):
	'''
	Return a list of all events found in the list of filenames passed in,
	where an event is represented by a list of segment means instead of the
	event object. This is primarily used for training.
	'''

	events = [] 
	for filename in filenames:
		json_name = filename[:-4]+".json"

		try:
			if force_parse:
				raise Error()

			file = File.from_json( json_name )
			for event in file.events:
				events.append( [ seg.mean for seg in event.segments ] )
			print "\t{} successfully read".format( json_name )
			file.close()

		except:
			print "\t{} unsuccessful. Parsing .abf file".format( json_name )
			file = parse_file( filename, hmm )
			for event in file.events:
				events.append( [ seg.mean for seg in event.segments ] )
			print "\tSaving file to {}".format( json_name )
			file.to_json( filename=json_name )
			file.close() 

	return events

def analyze_events( events, hmm ):
	"""
	Take in a list of events and create a dataframe of the data. 
	"""

	data = {}
	data['Filter Score'] = []
	data['C'] = []
	data['mC'] = []
	data['hmC'] = []
	data['X'] = []
	data['T'] = []
	data['CAT'] = []
	data['Soft Call'] = []

	tags = ( 'C', 'mC', 'hmC', 'X', 'T', 'CAT' )
	indices = { state.name: i for i, state in enumerate( hmm.states ) }

	for idx, event in enumerate( events ):
		# Hold data for the single event in 'd'. The fields will hold a single
		# piece of information.
		d = { key: None for key in data.keys() }

		# Run forward-backward on that event to get the expected transitions
		# matrix
		#print event
		trans, ems = hmm.forward_backward( event )

		# Get the expected number of transitions from all states to the first 
		# match state of each fork, and the the expected number of transitions
		# from the last match state to the closing of the fork.
		# The measurement we use for this is the minimum of the expected 
		# transitions into the fork and the expected transitions out of the
		# fork, because we care about the expectation of going through the
		# entire event.
		for tag in 'C', 'mC', 'hmC':
			names = [ "M-{}:{}-end".format( tag, i ) for i in xrange(25, 34) ]
			d[ tag ] = min( [ trans[ indices[name] ].sum() for name in names ] )

		# Perform the same calculation, except on the label fork now instead
		# of the cytosine variant fork.
		for tag in 'X', 'T', 'CAT':
			names = [ "M-{}:{}-end".format( tag, i ) for i in xrange(37, 43) ]
			d[ tag ] = min( [ trans[ indices[name] ].sum() for name in names ] )

		# Calculate the score, which will be the sum of all expected
		# transitions to each of the three forks, and the expected transitions
		# into each of the three labels. This gives us a score representing
		# how likely it is that this event went through the fork and through
		# the label.
		d['Filter Score'] = sum( d[tag] for tag in tags[:3] ) * sum( d[tag] for tag in tags[3:] )
 
		# Calculate the dot product score between the posterior transition
		# probability of transitions for cytosine variants, and the
		# corresponding labels
		score = d['C']*d['T'] + d['mC']*d['CAT'] + d['hmC']*d['X']
		d['Soft Call'] = score / d['Filter Score'] if d['Filter Score'] != 0 else 0

		for key, value in d.items():
			data[key].append( value )
	
	return pd.DataFrame( data )

def insert_delete_plot( model, events ):
	"""
	Calculate the probability of an insert or delete at each position in the
	HMM.
	"""

	# Get the length of the profile from the HMM name
	n = int( model.name.split('-')[-1] )

	# Get a mapping from states to indices in the state list
	indices = { state.name: i for i, state in enumerate( model.states ) }
	delete_names = [ state.name for state in model.states if 'D' in state.name ]
	insert_names = [ state.name for state in model.states if 'I' in state.name ]
	backslip_names = [ state.name for state in model.states if 'b' in state.name and state.name[-2:] == 'e7' ]
	underseg_names = [ state.name for state in model.states if 'U' in state.name and 's' not in state.name and 'e' not in state.name ]
	array_names = [ delete_names, insert_names, backslip_names, underseg_names ]

	# Create a list of expected transitions to deletes and inserts respectively
	deletes = np.zeros( n+1 ) 
	inserts = np.zeros( n+1 )
	backslips = np.zeros( n+1 )
	undersegmentation = np.zeros( n+1 )
	arrays = [ deletes, inserts, backslips, undersegmentation ]

	# Go through each event, calculating the number of transitions
	for event in events:
		# Run forward-backward to get the expected transition counts
		trans, ems = model.forward_backward( event )

		# For each delete state, add the number of expected transitions
		# for this event to the total running count
		for array, names in zip( arrays, array_names ):
			for name in names:
				if array is backslips:
					position = int( name.split(":")[-1].split("e")[0] )
				else:
					position = int( name.split(":")[-1] )
				index = indices[ name ]

				array[ position ] += trans[ :, index ].sum()

	for array in arrays:
		array /= float( len(events) )

	plt.subplot(411)
	plt.bar( np.arange(n+1)+0.2, inserts, 0.8, color='c', alpha=0.66 )

	plt.subplot(412)
	plt.bar( np.arange(n+1)+0.2, deletes, 0.8, color='m', alpha=0.66 )

	plt.subplot(413)
	plt.bar( np.arange(n+1)+0.2, backslips, 0.8, color='#FF6600', alpha=0.66 )

	plt.subplot(414)
	plt.bar( np.arange(n+1)+0.2, undersegmentation, 0.8, color='g', alpha=0.66 )

	plt.show()

def train( model, events, threshold=0.10 ):
	'''
	This is a full step of training of the model. This involves a cross-training
	step to determine which parameter should be used to filter out events from
	the full training step, and then 10 iterations of Baum-Welch training. This
	assumes that all events passed in are training events.
	'''

	tic = time.time()
	data = analyze_events( events, model )
	print "Scoring Events Took {}s".format( time.time() - tic )

	# Get the events using the best threshold
	events = [ event for score, event in it.izip( data['Filter Score'], events ) if score > threshold ]
	print "{} events after filtering".format( len(events) )

	# Train the HMM using those events
	tic = time.time()
	model.train( events, max_iterations=10, use_pseudocount=True )
	print "Training on {} events took {}".format( len( events ), time.time() - tic )

	return model

def test( model, events ):
	'''
	Test the model on the events, returning a list of accuracies should the top
	i events be used.
	'''

	# Analyze the data, and sort it by filter score
	data = analyze_events( events, model ).sort( 'Filter Score' )
	n = len(data)

	# Attach the list of accuracies using the top i events to the frame
	data['MCSC'] = [ 1. * sum( data['Soft Call'][i:] ) / (n-i) for i in xrange( n ) ]

	# Return the frame
	return data

def n_fold_cross_validation( events, n=5 ):
	'''
	Perform n-fold cross-validation, wherein the data is split into n equally
	sized sections, and the model is trained on all but one of them, and
	classifies the last one. This is repeated until each section has been
	classified. For each iteration, the model is pulled fresh from a text
	document to ensure that it is not modified by any round of training.
	'''

	# Divide the data into n equally sized folds
	folds = [ events[i::n] for i in xrange( n ) ]

	# Go through each one, defining which one is for testing and which ones
	# are for training.
	for i in xrange( n ):
		training = reduce( lambda x,y: x+y, folds[:i] + folds[i+1:], [] )
		testing = folds[i]

		with open( 'untrained_hmm.txt', 'r' ) as infile:
			model = Model.read( infile )
		model = train( model, training, threshold=0.1 )

		if i == 0:
			data = test( model, testing )
		else:
			data = pd.concat( [ data, test(model, testing) ] )

	data = data.sort( 'Filter Score' )
	n = len(data)

	return [ 1. * sum( data['Soft Call'][i:] ) / (n-i) for i in xrange( n ) ][::-1]

def train_test_split( events, train_perc ):
	'''
	Takes in a set of events, and splits it into a training and a testing
	split, where the training set consists of train_perc*100% of the data.
	'''

	return events[ :int( len(events) * train_perc ) ], events[ int( len(events) * train_perc ): ]

def threshold_scan( train, test ):
	'''
	When training is done, a threshold is set on the filter score of the events
	used for training. This will scan a range of these scores, and give the
	accuracy as to each one.
	'''

	with open( 'untrained_hmm.txt', 'r' ) as infile:
		model = Model.read( infile )

	# Get the filter scores for each event in the training set
	train_data = analyze_events( train, model )

	# Store the accuracies in a list
	accuracies = []

	# Scan through a range of thresholds... 
	for threshold in 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9:
		print "Threshold set at {}".format( threshold )

		# Take only the events whose filter socre is above a threshold
		event_subset = [ event for event, score in it.izip( train, train_data.Score ) if score > threshold ]

		# If no events left, do not perform training.
		if len(event_subset) == 0:
			continue

		# Open a fresh copy of the HMM.
		with open( 'untrained_hmm.txt', 'r' ) as infile:
			model = Model.read( infile )
		
		# Train the model on the training events
		model.train( event_subset, max_iterations=10, use_pseudocount=True )

		# Now score each of the testing events
		data = analyze_events( test, model ).sort( 'Score' )
		n = len(data)

		# Attach the list of accuracies using the top i events to the frame
		accuracies.append( [ 1. * sum( data['Soft Call'][i:] ) / (n-i) for i in xrange( n ) ][::-1] )

	# Turn this list into a numpy array for downstream use
	accuracies = np.array( accuracies )

	return accuracies


"""	"""	"""	"""
"""	"""	"""	"""

if __name__ == '__main__':
	# List all the files which will be used in the analysis.
	files = [ '14418004-s04.abf', '14418005-s04.abf', '14418006-s04.abf',
			  '14418007-s04.abf', '14418008-s04.abf', '14418009-s04.abf', 
			  '14418010-s04.abf', '14418011-s04.abf', '14418012-s04.abf', 
			  '14418013-s04.abf', '14418014-s04.abf', '14418015-s04.abf', 
			  '14418016-s04.abf' ]

	print "Beginning"
	#print "Building Profile HMM..."

	with open( "untrained_hmm.txt") as infile:
		model = Model.read( infile )

	# Get all the events
	events = get_events( files, model )
	print "{} events detected".format( len(events) )

	train_fold, test_fold = train_test_split( events, train_perc=0.7 )
	print "{} Training Events and {} Testing Events".format( len(train_fold), len(test_fold) )

	model = train( model, train_fold, threshold=0.1 )
	data = test( model, test_fold )

	data.to_csv( 'Test Set.csv' )


	import random
	# Cross Validation
	accuracies = []
	for i in xrange( 10 ):
		print "\nIteration {}".format( i )
		random.shuffle( events )
		accuracies.append( n_fold_cross_validation( events, n=5 ) )

	accuracies = np.array( accuracies )
	np.savetxt( 'n_fold_accuracies.txt', accuracies )

	plt.plot( accuracies.mean(axis=0) )
	plt.show()

	sys.exit()
