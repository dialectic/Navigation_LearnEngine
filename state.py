import numpy as np
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Wedge, Polygon, Arc, RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.transforms import BboxBase, Bbox
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
cmaps = OrderedDict()
from matplotlib.legend_handler import HandlerPatch

class State:
	"""
	A class that represents a state in
	some coordinate system both as coordinates
	and as fuzzy tags. Initialize with :py:attr:`x` or :py:attr:`X`. This is an
	'in-place' class that overwrites itself.
	Each body's state is defined with a
	sub-list of :py:attr:`x`.

	:param x: coordinate lists of state of *n* bodies (see attribute :py:attr:`x`)
	:type x: :py:class:`list` of :py:class:`list`'s of 6 coordinates :math:`[x,y,θ,\dot{x},\dot{y},\dot{θ}]`
	
	:param X: fuzzy tag dicts of state (see attribute :py:attr:`X`)
	:type X: :py:class:`list` of :py:class:`dict`'s of tags with keys tag names and values fuzzy membership values
	
	:param coord_sys: name of coordinate system. Has no effect other than denotation.
	:type coord_sys: :py:class:`str`, optional

	:param th: thresholds for fuzzy membership in form ``{'p_lo': 1,'p_hi': 2,...}``. These should all be nonnegative.
		
		- ``p_lo``, ``p_hi`` -- distance thresholds for distance/position state variables :math:`x,y`.
		- ``v_lo``, ``v_hi`` -- speed thresholds for speed state variables :math:`\dot{x},\dot{y}`.
		- ``a_lo``, ``a_hi`` -- angular thresholds for angular distance state :math:`θ`.
		- ``as_lo``, ``as_hi`` -- angular speed thresholds for angular speed state :math:`\dot{θ}`.

	:type th: :py:class:`dict`, optional

	Any coordinate system can be used as long as
	it complies with the following form.

	The class is for planar (two-dimensional) rectangular coordinates.
	They can be either fixed/inertial or moving/body-fixed.
	
	Any number :math:`n` of bodies can be described in an instance.
	Each body must have the following state vector :math:`\\boldsymbol{x}`:

	.. math::
		:label: state-vector

		\\boldsymbol{x} = 
		\\begin{bmatrix}
			x & y & θ & \\dot{x} & \\dot{y} & \\dot{θ}
		\\end{bmatrix}^\\top

	The states of :eq:`state-vector` are defined as follows:

		- :math:`x`, :math:`y` -- position coordinates
		- :math:`\\theta` -- angular position coordinate (from :math:`+x`)
		- :math:`\\dot{x}`, :math:`\\dot{y}` -- velocities in the :math:`x` and :math:`y` directions
		- :math:`\\dot{\\theta}` -- angular velocity coordinate.

	Fuzzy sets/tags/categories give an alternative representation
	of the state as :py:attr:`X`, a list of these categories with
	corresponding fuzzy set membership values in the interval :math:`[0,1]`.
	This representation is computed automatically when constructing
	an instance of :py:class:`State` with coordinates :py:attr:`x`.
	The following figure shows an overview of the fuzzy
	categories provided.

	.. _membership:
	.. figure:: figures/dia-navigation-membership.svg

		An illustration of coordinates and corresponding
		categories, 10 in total. The "vehicle" is shown
		to illustrate the qualitative sense of the categories.
		When the coordinate system is body-fixed to a vehicle,
		the states of other vehicles will be described 
		in categories, that is, qualitatively, relative to that
		vehicle. The underlined letter corresponds to the :py:class:`State`
		method instantiating the corresponding membership function.

	Membership functions allow the membership values of
	the categories :py:attr:`X` to be computed from the state :py:attr:`x`.
	The membership functions are instantiated in :py:class:`State`
	methods correspond in name to :numref:`membership`. Specifically,

		- distance right :math:`\\underline{R}`: membership function method :py:meth:`R`
		- distance left :math:`\\underline{L}`: membership function method :py:meth:`L`
		- distance back :math:`\\underline{B}`: membership function method :py:meth:`B`
		- distance forth :math:`\\underline{F}`: membership function method :py:meth:`F`
		- speed right :math:`\\underline{\\dot{R}}`: membership function method :py:meth:`Rp`
		- speed left :math:`\\underline{\\dot{L}}`: membership function method :py:meth:`Lp`
		- speed back :math:`\\underline{\\dot{B}}`: membership function method :py:meth:`Bp`
		- speed forth :math:`\\underline{\\dot{F}}`: membership function method :py:meth:`Fp`
		- angle left :math:`\\underline{λ}`: membership function method :py:meth:`λ`
		- angle right :math:`\\underline{ρ}`: membership function method :py:meth:`ρ`
		- angular speed left :math:`\\underline{\\dot{λ}}`: membership function method :py:meth:`λp`
		- angular right :math:`\\underline{\\dot{ρ}}`: membership function method :py:meth:`ρp`

	When a :py:class:`State` instance is constructed with 
	states via :py:attr:`x`, the corresponding categories/tags
	of :py:attr:`X` are automatically computed and populated
	via the method :py:meth:`tag`.

	.. code-block:: python

		s1 = [1,-3,0,1,0,-1] # state of body 1
		s2 = [0,2,-1,0,-1,1] # state of body 2
		s = State(x=[s1,s2]) # instantiate with default params
		# the categories/tags of s.X are automatically computed
		print(s.X) # print categories/tags representation

	Alternatively, when a :py:class:`State` instance is constructed with 
	tags/categories via :py:attr:`X`, the corresponding states
	of :py:attr:`x` are automatically computed and populated 
	via the method :py:meth:`coordinate`. 

	.. code-block:: python

		s1 = { # state of body 1
			'L': 0, 'R': 0,
			'F': 0, 'B': 0, 
			'λ': 0, 'ρ': 0,
			'Lp': 0, 'Rp': 0.4,
			'Fp': 0.3, 'Bp': 0,
			'λp': 0.1, 'ρp': 0
		}
		s2 = { # state of body 2
			'L': 0, 'R': 0.4,
			'F': 0.2, 'B': 0,
			'λ': 0, 'ρ': 0.5,
			'Lp': 0.1, 'Rp': 0,
			'Fp': 0, 'Bp': 1,
			'λp': 0, 'ρp': 0.3
		}
		s = State(X=[s1,s2]) # instantiate with default params
		# the state of s.x is automatically computed
		print(s.x) # print state/coordinate representation

	.. note:: In this example, ``s1`` has all distance and angle 
		tags set to ``0``. This corresponds with a coordinate 
		system that is body-fixed to "body 1".
	
	The :py:meth:`coordinate` method uses
	inverse membership functions explicitly defined as methods with the 
	name of the non-inverse membership function with the suffix
	``_inv``, e.g. :py:meth:`L_inv`. The membership functions
	are non-surjective and therefore non-bijective (non-invertible)
	so the "inverse" functions are "lossy" (not true inverses).

	Helper methods such as :py:meth:`s_curve` and :py:meth:`speed_membership`
	make uniform the membership function and inverse methods.
	Plotting methods include :py:meth:`plot_state` and 
	:py:meth:`plot_s_curve`.
	"""

	def __init__(self,x=None,X=None,coord_sys=None,th=None):
		if not th:
			self.th = { #: default membership thresholds
				'p_lo': 1,
				'p_hi': 10,
				'v_lo': 0.1,
				'v_hi': 2,
				'a_lo': np.pi/12,
				'a_hi': 11*np.pi/12,
				'as_lo': np.pi/24,
				'as_hi': 23*np.pi/24
			}
		if x:
			if not isinstance(x[0],list):
				self.x = [x]
			else:
				#: list: :py:class:`list` of :py:class:`list`'s of 6 coordinates :math:`[x,y,θ,\dot{x},\dot{y},\dot{θ}]`
				self.x = x
			self.tag() # propagate
		elif X:
			#: list: :py:class:`list` of :py:class:`dict`'s of tags with keys tag names and values fuzzy membership values
			self.X = X
			self.coordinate() # propagate
		else:
			raise(Exception('must specify either x or X'))
		
		if coord_sys:
			self.coord_sys = coord_sys
		else:
			warnings.warn('No coord_sys specified')

	def tag(self):
		"""A method to compute the fuzzy tag
		membership values and populate X.
		"""
		n = len(self.x) # num of bodies
		# generate tags for each body
		tag_bases = ['L','F','B','R']+\
		['Lp','Fp','Bp','Rp']+\
		['λ','ρ','λp','ρp']
		tag_ll = [tag_bases]*n
		X = []
		for i,tag_obj in enumerate(tag_ll):
			X.append({})
			for tag in tag_obj:
				try:
					X[-1][tag] = eval(f'self.{tag}')(self.x[i])
				except Exception as e:
					raise
		self.X = X

	def coordinate(self):
		"""A method to compute the inverse
		fuzzy tag membership values and
		populate x.
		"""
		n = len(self.X) # num of bodies
		X = self.X # memberships
		x = []
		for b in range(0,n): # n bodies
			xb = []
			for c in range(0,6): # 6 coordinates
				if c == 0: # x
					if self.F_inv(X[b]):
						xb.append(self.F_inv(X[b]))
					elif self.B_inv(X[b]):
						xb.append(self.B_inv(X[b]))
					else:
						xb.append(0)
				elif c == 1: # y
					if self.L_inv(X[b]):
						xb.append(self.L_inv(X[b]))
					elif self.R_inv(X[b]):
						xb.append(self.R_inv(X[b]))
					else:
						xb.append(0)
				elif c == 2: # θ
					if self.λ_inv(X[b]):
						xb.append(self.λ_inv(X[b]))
					elif self.ρ_inv(X[b]):
						xb.append(self.ρ_inv(X[b]))
					else:
						xb.append(0) # Forward
				elif c == 3: # x'
					if self.Fp_inv(X[b]):
						xb.append(self.Fp_inv(X[b]))
					elif self.Bp_inv(X[b]):
						xb.append(self.Bp_inv(X[b]))
					else:
						xb.append(0)
				elif c == 4: # y'
					if self.Lp_inv(X[b]):
						xb.append(self.Lp_inv(X[b]))
					elif self.Rp_inv(X[b]):
						xb.append(self.Rp_inv(X[b]))
					else:
						xb.append(0)
				elif c == 5: # θ'
					if self.λp_inv(X[b]):
						xb.append(self.λp_inv(X[b]))
					elif self.ρp_inv(X[b]):
						xb.append(self.ρp_inv(X[b]))
					else:
						xb.append(0) # Forward
			x.append(xb)
		self.x = x # update instance

	def L(self,x_b):
		"""Membership function for left L of
		a body with state ``x_b``. L ~ +y.

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		if x_b[1] <= 0:
			return 0
		else:
			return self.position_membership(x_b[1])

	def L_inv(self,X_b):
		"""Inverse membership function for 
		left L of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['L'] <= 0:
			return None
		elif X_b['L'] >= 1:
			return self.th['p_hi']
		else:
			return self.position_membership_inverse(X_b['L'])

	def F(self,x_b):
		"""Membership function for forward F of
		a body with state ``x_b``. F ~ +x

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		if x_b[0] <= 0:
			return 0
		else:
			return self.position_membership(x_b[0])

	def F_inv(self,X_b):
		"""Inverse membership function for 
		forward F of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['F'] <= 0:
			return None
		elif X_b['F'] >= 1:
			return self.th['p_hi']
		else:
			return self.position_membership_inverse(X_b['F'])

	def B(self,x_b):
		"""Membership function for back B of
		a body with state ``x_b``.

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		if x_b[0] >= 0:
			return 0
		else:
			return self.position_membership(abs(x_b[0]))

	def B_inv(self,X_b):
		"""Inverse membership function for 
		back B of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['B'] <= 0:
			return None
		elif X_b['B'] >= 1:
			return -1*self.th['p_hi']
		else:
			return -1*self.position_membership_inverse(X_b['B'])

	def R(self,x_b):
		"""Membership function for right R of
		a body with state ``x_b``. R ~ -y

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		if x_b[1] >= 0:
			return 0
		else:
			return self.position_membership(abs(x_b[1]))

	def R_inv(self,X_b):
		"""Inverse membership function for 
		right R of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['R'] <= 0:
			return None
		elif X_b['R'] >= 1:
			return -1*self.th['p_hi']
		else:
			return -1*self.position_membership_inverse(X_b['R'])

	def Lp(self,x_b):
		"""Membership function for speed left L of
		a body with state ``x_b``. Lp ~ +y'

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		if x_b[4] <= 0:
			return 0
		else:
			return self.speed_membership(x_b[4])

	def Lp_inv(self,X_b):
		"""Inverse membership function for 
		speed left L of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['Lp'] <= 0:
			return None
		elif X_b['Lp'] >= 1:
			return self.th['v_hi']
		else:
			return self.speed_membership_inverse(X_b['Lp'])

	def Fp(self,x_b):
		"""Membership function for speed forward F of
		a body with state ``x_b``. Fp ~ +x'

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		if x_b[3] <= 0:
			return 0
		else:
			return self.speed_membership(x_b[3])

	def Fp_inv(self,X_b):
		"""Inverse membership function for 
		speed forward F of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['Fp'] <= 0:
			return None
		elif X_b['Fp'] >= 1:
			return self.th['v_hi']
		else:
			return self.speed_membership_inverse(X_b['Fp'])

	def Bp(self,x_b):
		"""Membership function for speed back B of
		a body with state ``x_b``. Bp ~ -x'

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		if x_b[3] >= 0:
			return 0
		else:
			return self.speed_membership(abs(x_b[3]))

	def Bp_inv(self,X_b):
		"""Inverse membership function for 
		speed back B of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['Bp'] <= 0:
			return None
		elif X_b['Bp'] >= 1:
			return -1*self.th['v_hi']
		else:
			return -1*self.speed_membership_inverse(X_b['Bp'])

	def Rp(self,x_b):
		"""Membership function for speed right R of
		a body with state ``x_b``. Rp ~ -y'

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		if x_b[4] >= 0:
			return 0
		else:
			return self.speed_membership(abs(x_b[4]))

	def Rp_inv(self,X_b):
		"""Inverse membership function for 
		speed right R of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['Rp'] <= 0:
			return None
		elif X_b['Rp'] >= 1:
			return -1*self.th['v_hi']
		else:
			return -1*self.speed_membership_inverse(X_b['Rp'])

	def λ(self,x_b):
		"""Membership function for angle left of
		a body with state ``x_b``.

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		φ = self.angle_normalizer(x_b[2])
		if φ <= 0:
			return 0
		else:
			return self.angle_membership(φ)

	def λ_inv(self,X_b):
		"""Inverse membership function for 
		angle left of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['λ'] <= 0:
			return None
		elif X_b['λ'] >= 1:
			return self.th['a_hi']
		else:
			return self.angle_membership_inverse(X_b['λ'])

	def ρ(self,x_b):
		"""Membership function for angle right of
		a body with state ``x_b``.

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		φ = self.angle_normalizer(x_b[2])
		if φ >= 0:
			return 0
		else:
			return self.angle_membership(abs(φ))

	def ρ_inv(self,X_b):
		"""Inverse membership function for 
		angle right of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['ρ'] <= 0:
			return None
		elif X_b['ρ'] >= 1:
			return -1*self.th['a_hi']
		else:
			return -1*self.angle_membership_inverse(X_b['ρ'])

	def λp(self,x_b):
		"""Membership function for angle speed left of
		a body with state ``x_b``.

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		φp = x_b[5]
		if φp <= 0:
			return 0
		else:
			return self.angle_speed_membership(φp)

	def λp_inv(self,X_b):
		"""Inverse membership function for 
		angle speed left of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['λp'] <= 0:
			return None
		elif X_b['λp'] >= 1:
			return self.th['as_hi']
		else:
			return self.angle_speed_membership_inverse(X_b['λp'])

	def ρp(self,x_b):
		"""Membership function for angle speed right of
		a body with state ``x_b``.

		:param x_b: the coordinate state of a body
		:type x_b: :py:class:`list`

		:returns: membership value, :py:class:`float`
		"""
		φp = x_b[5]
		if φp >= 0:
			return 0
		else:
			return self.angle_speed_membership(abs(φp))

	def ρp_inv(self,X_b):
		"""Inverse membership function for 
		angle speed right of a body with membership ``X_b``.

		:param X_b: categorical membership state of a body in the form of `key: value` pairs corresponding to `<tag>: <membership value>` 
		:type X_b: :py:class:`dict`

		:returns: coordinate state of the body, :py:class:`list`
		"""
		if X_b['ρp'] <= 0:
			return None
		elif X_b['ρp'] >= 1:
			return -1*self.th['as_hi']
		else:
			return -1*self.angle_speed_membership_inverse(X_b['ρp'])

	def position_membership(self,p):
		"""Position membership function used
		for all position tags. Arg p is the
		coordinate x or y. Returns membership
		value in interval [0,1].
		"""
		return self.s_curve(p,self.th['p_lo'],self.th['p_hi'])

	def position_membership_inverse(self,m):
		"""Inverse position membership function used
		for all position tags. Arg m is the
		membership value. Returns coordinate x or y
		value in interval [p_lo,p_hi].
		"""
		return self.s_curve_inverse(m,self.th['p_lo'],self.th['p_hi'])

	def speed_membership(self,v):
		"""Speed membership function used
		for all speed tags. Arg `v` is the
		coordinate speed `|x'|` or `|y'|`.
		Returns membership value in interval `[0,1]`.
		"""
		return self.s_curve(v,self.th['v_lo'],self.th['v_hi'])

	def speed_membership_inverse(self,m):
		"""Inverse speed membership function used
		for all speed tags. Arg `m` is the
		membership value.
		Returns speed `|x'|` or `|y'|` in `[0,v_hi]`.
		"""
		return self.s_curve_inverse(m,self.th['v_lo'],self.th['v_hi'])

	def angle_membership(self,a):
		"""Angle membership function used
		for all angle tags. Arg `a` is the
		angle `λ` or `ρ`. Returns membership
		value in interval `[0,1]`.
		"""
		return self.s_curve(a,self.th['a_lo'],self.th['a_hi'])

	def angle_membership_inverse(self,m):
		"""Inverse angle membership function used
		for all angle tags. Arg `m` is the
		angle. Returns angle `λ` or `ρ` coordinate
		value in interval `[0,a_hi]`.
		"""
		return self.s_curve_inverse(m,self.th['a_lo'],self.th['a_hi'])

	def angle_speed_membership(self,a):
		"""Angle speed membership function used
		for all angular speed tags. Arg `a` is the
		angular speed `λp` or `ρp`. Returns membership
		value in interval `[0,1]`.
		"""
		return self.s_curve(a,self.th['as_lo'],self.th['as_hi'])

	def angle_speed_membership_inverse(self,a):
		"""Inverse angle speed membership function used
		for all angular speed tags. Arg `m` is the
		angular speed membership. Returns angular speed
		`λp` or `ρp` value in interval `[0,1]`.
		"""
		return self.s_curve_inverse(a,self.th['as_lo'],self.th['as_hi'])

	def angle_normalizer(self,a):
		"""Normalizes angle a to -pi,pi
		with a=0 in Forward direction."""
		a = a % (2*np.pi)
		if a < -np.pi:
			a += 2*np.pi
		elif a > np.pi:
			a -= 2*np.pi
		return a # 0 is Forward

	def angle_denormalizer(self,a):
		"""Denormalizes angle a to -pi,pi
		with a=0 in Forward direction."""
		return a # 0 is Forward

	def s_curve(self,p,p_lo,p_hi):
		"""s-curve (lossy) function for 
		membership values from coordinate
		values.
		"""
		if p <= p_lo:
			return 0
		elif p >= p_hi:
			return 1
		else:
			return 1/2-np.cos(np.pi*(p-p_lo)/(p_hi-p_lo))/2

	def s_curve_inverse(self,m,p_lo,p_hi):
		"""Inverse s-curve (lossy) for
		coordinate values from membership
		values.
		"""
		if m == 1:
			return p_hi
		elif m == 0:
			return p_lo
		else:
			return np.arccos(1-2*m)*(p_hi-p_lo)/np.pi+p_lo

	def plot_s_curve(self):
		"""Plot the s-curve helper method for membership functions"""
		p_lo = 1
		p_hi = 7
		x = np.linspace(0,p_hi*1.1,100)
		y = list(map(lambda a: self.s_curve(a,p_lo,p_hi),x))
		plt.plot(x,y)
		plt.xticks([0,p_lo,p_hi],['0','lo','hi'])
		plt.show()

	def plot_s_curve_inverse(self):
		"""Plot the inverse s-curve helper method for inverse membership functions"""
		p_lo = 1
		p_hi = 7
		x = np.linspace(0,1,100)
		y = list(map(lambda m: self.s_curve_inverse(m,p_lo,p_hi),x))
		plt.plot(x,y)
		plt.yticks([0,p_lo,p_hi],['0','lo','hi'])
		plt.show()

	def plot_state(self,save=False,fname=None):
		"""Plot the state.

		:param save: To save as pdf and svg or not
		:type save: :py:class:`bool`, optional

		:param fname: Filename of saved file (stem, no extension)
		:type fname: :py:class:`str`, optional

		:returns: The figure handle

		Example:

		.. code-block:: python

			x1 = State(x=[[3,-2,np.pi/3,3,0,2],[-7,5,np.pi/2,-1,4,-1]],coord_sys='a')
			x1.plot_state()
		
		The resulting figure is shown in :numref:`plot_state_example`.

		.. _plot_state_example:
		.. figure:: figures/playground_02.svg

			An example of a two-body system's state plotted
			with :meth:`plot_state`.

		"""
		fig, ax = plt.subplots()
		fig.tight_layout()
		ax.grid(zorder=-1)
		ax.set_aspect("equal", adjustable="datalim")
		ax.plot(-1,-1) # keeps origin in view
		ax.plot(1,1) # keeps origin in view
		X = [];Y = [];U = [];V = []
		for i,body in enumerate(self.x):
			ax.scatter(body[1],body[0],zorder=13,color='k',marker='.')
			X.append(body[1])
			Y.append(body[0])
			U.append([-body[4]])
			V.append([body[3]])
		quiv = ax.quiver(X,Y,U,V,zorder=12,capstyle='round',joinstyle='round',alpha=.5)
		fig.canvas.draw() # to get proper lims/bounds
		lim_tuple_x = ax.get_xbound()
		lim_tuple_y = ax.get_ybound()
		body_length_x = abs(lim_tuple_x[1]-lim_tuple_x[0])/12
		body_length_y = abs(lim_tuple_y[1]-lim_tuple_y[0])/12
		body_length = max(body_length_x,body_length_y)
		patches = []
		leg = []
		da = 20
		for i,body in enumerate(self.x):
			a = np.rad2deg(-body[2]-np.pi/2)
			dx = 2*body_length*np.cos(np.deg2rad(a))/3
			dy = 2*body_length*np.sin(np.deg2rad(a))/3
			patches += [Wedge(
				(body[1]-dx,body[0]-dy),
				body_length,a-da,a+da,
				facecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][i+6],
				edgecolor='k',
				linewidth=.5,
				clip_on=False,
				zorder=10,
				alpha=1,
				label=f'body {i}'
			)]
			angular_speed_is = True
			if body[5] > 0:
				reverse_dir = True
			elif body[5] < 0:
				reverse_dir = False
			else:
				angular_speed_is = False
			if angular_speed_is:
				self.drawCirc(ax,
					2*body_length,
					body[1],body[0],
					180+a-2*da,90,
					color_= cm.coolwarm_r(
						(np.sign(body[5])*self.λp(np.abs(body)))/2+1/2
					),
					reverse_dir=reverse_dir,
				)
		ax.set_xlabel("y",loc="left")
		ax.set_ylabel("x",loc="top")
		p = PatchCollection(patches,clip_on=False,match_original=True,zorder=10)
		ax.add_collection(p)
		ax.invert_xaxis()
		ax.spines['left'].set_position(('data', 0.0))
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_position(('data', 0.0))
		ax.spines['top'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		marg = ax.margins()
		ax.margins(1.25*marg[0],1.25*marg[1])
		ax.legend(
			handles=patches,
			handler_map={Wedge: self.HandlerWedge()},
			bbox_to_anchor=(1,-0.04), loc="upper right"
		)
		position=fig.add_axes([.125,0,.5,.05])
		cbar = fig.colorbar(
			cm.ScalarMappable(
				norm=mpl.colors.Normalize(vmin=-1, vmax=1), 
				cmap='coolwarm'
			), 
			ax=ax,
			label=r'angular velocity $\omega$',
			orientation="horizontal",
			shrink=.5,
			ticks=[-1,0,1],
			cax=position,
			# anchor=(0,0)
		)
		cbar.ax.set_xticklabels([r'$+$', '', r'$-$'])  # horizontal colorbar
		plt.show()
		if save:
			if not fname:
				raise(Exception('Must provide filename to save'))
			else:
				fig.savefig(fname+'.pdf',format='pdf',
					transparent=True,
					bbox_inches='tight'
				)
				fig.savefig(fname+'.svg',format='svg',
					transparent=True,
					bbox_inches='tight'
				)
		return fig

	def drawCirc(self,ax,radius,centX,centY,angle_,theta2_,color_='black',reverse_dir=False):
	    arc = Arc((centX,centY),radius,radius,angle=angle_,
	          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=3,color=color_,clip_on=False)
	    ax.add_patch(arc)
	    if reverse_dir:
	    	orientation = angle_
	    else:
	    	orientation = theta2_+angle_
	    endX=centX+(radius/2)*np.cos(np.deg2rad(orientation)) #Do trig to determine end position
	    endY=centY+(radius/2)*np.sin(np.deg2rad(orientation))
	    if reverse_dir:
	    	orientation += 60
	    ax.add_patch(
	        RegularPolygon( #Create triangle as arrow head
	            (endX, endY),            # (x,y)
	            3,                       # number of vertices
	            radius/9,                # radius
	            np.deg2rad(orientation),     # orientation
	            color=color_
	        )
        )

	class HandlerWedge(HandlerPatch):
	    def create_artists(self, legend, orig_handle,
	                       xdescent, ydescent, width, height, fontsize, trans):
	        center = 0.3 * width - 0.1 * xdescent, 0.5 * height - 0.5 * ydescent
	        p = Wedge(center,width-xdescent,theta1=-20,theta2=20)
	        self.update_prop(p, orig_handle, legend)
	        p.set_transform(trans)
	        return [p]

class StateTrajectory:
	def __init__(self,states=[],dt=None):
		self.states = states
		if dt:
			self.dt = dt

	def plot_states(self,save=False,fname=None,nplot=10,decimation_method='linear'):
		"""Plot the state trajectory.

		:param save: To save as pdf and svg or not
		:type save: :py:class:`bool`, optional

		:param fname: Filename of saved file (stem, no extension)
		:type fname: :py:class:`str`, optional

		:returns: The figure handle

		Example:

		.. code-block:: python

			TODO
		
		The resulting figure is shown in :numref:`plot_state_example`.

		.. _plot_state_example:
		.. figure:: figures/playground_02.svg

			An example of a two-body system's state plotted
			with :meth:`plot_states`.

		"""
		# decimate
		decimation = int(np.floor(len(self.states)/nplot))
		if decimation == 0:
			decimation = 1
		if decimation_method == 'linear':
			idec = list(np.floor(np.linspace(0,len(self.states)-1,nplot)).astype(int))
		elif decimation_method == 'log':
			idec = list(np.floor(np.logspace(0,np.log10(len(self.states))-1,nplot)).astype(int))
		states_decimated = [self.states[i] for i in idec]
		fig, ax = plt.subplots()
		fig.tight_layout()
		ax.grid(zorder=-1)
		ax.set_aspect("equal", adjustable="datalim")
		ax.plot(-1,-1) # keeps origin in view
		ax.plot(1,1) # keeps origin in view
		X = [];Y = [];U = [];V = []
		for j,state in enumerate(states_decimated):
			for i,body in enumerate(state.x):
				ax.scatter(body[1],body[0],zorder=13,color='k',marker='.')
				X.append(body[1])
				Y.append(body[0])
				U.append([-body[4]])
				V.append([body[3]])
		quiv = ax.quiver(X,Y,U,V,zorder=12,capstyle='round',joinstyle='round',alpha=.5)
		fig.canvas.draw() # to get proper lims/bounds
		lim_tuple_x = ax.get_xbound()
		lim_tuple_y = ax.get_ybound()
		body_length_x = abs(lim_tuple_x[1]-lim_tuple_x[0])/12
		body_length_y = abs(lim_tuple_y[1]-lim_tuple_y[0])/12
		body_length = max(body_length_x,body_length_y)
		patches = []
		leg = []
		da = 20
		for j,state in enumerate(states_decimated):
			for i,body in enumerate(state.x):
				a = np.rad2deg(-body[2]-np.pi/2)
				dx = 2*body_length*np.cos(np.deg2rad(a))/3
				dy = 2*body_length*np.sin(np.deg2rad(a))/3
				patches += [Wedge(
					(body[1]-dx,body[0]-dy),
					body_length,a-da,a+da,
					facecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][i+6],
					edgecolor='k',
					linewidth=.5,
					clip_on=False,
					zorder=10,
					alpha=1,
					label=f'body {i}'
				)]
				angular_speed_is = True
				if body[5] > 0:
					reverse_dir = True
				elif body[5] < 0:
					reverse_dir = False
				else:
					angular_speed_is = False
				if angular_speed_is:
					self.drawCirc(ax,
						2*body_length,
						body[1],body[0],
						180+a-2*da,90,
						color_= cm.coolwarm_r(
							(np.sign(body[5])*state.λp(np.abs(body)))/2+1/2
						),
						reverse_dir=reverse_dir,
					)
		ax.set_xlabel("y",loc="left")
		ax.set_ylabel("x",loc="top")
		p = PatchCollection(patches,clip_on=False,match_original=True,zorder=10)
		ax.add_collection(p)
		ax.invert_xaxis()
		ax.spines['left'].set_position(('data', 0.0))
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_position(('data', 0.0))
		ax.spines['top'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		marg = ax.margins()
		ax.margins(1.25*marg[0],1.25*marg[1])
		position=fig.add_axes([.125,0,.5,.05])
		cbar = fig.colorbar(
			cm.ScalarMappable(
				norm=mpl.colors.Normalize(vmin=-1, vmax=1), 
				cmap='coolwarm'
			), 
			ax=ax,
			label=r'angular velocity $\omega$',
			orientation="horizontal",
			shrink=.5,
			ticks=[-1,0,1],
			cax=position,
			# anchor=(0,0)
		)
		cbar.ax.set_xticklabels([r'$+$', '', r'$-$'])  # horizontal colorbar
		plt.show()
		if save:
			if not fname:
				raise(Exception('Must provide filename to save'))
			else:
				fig.savefig(fname+'.pdf',format='pdf',
					transparent=True,
					bbox_inches='tight'
				)
				fig.savefig(fname+'.svg',format='svg',
					transparent=True,
					bbox_inches='tight'
				)
		return fig

	def drawCirc(self,ax,radius,centX,centY,angle_,theta2_,color_='black',reverse_dir=False):
	    arc = Arc((centX,centY),radius,radius,angle=angle_,
	          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=3,color=color_,clip_on=False)
	    ax.add_patch(arc)
	    if reverse_dir:
	    	orientation = angle_
	    else:
	    	orientation = theta2_+angle_
	    endX=centX+(radius/2)*np.cos(np.deg2rad(orientation)) #Do trig to determine end position
	    endY=centY+(radius/2)*np.sin(np.deg2rad(orientation))
	    if reverse_dir:
	    	orientation += 60
	    ax.add_patch(
	        RegularPolygon( #Create triangle as arrow head
	            (endX, endY),            # (x,y)
	            3,                       # number of vertices
	            radius/9,                # radius
	            np.deg2rad(orientation),     # orientation
	            color=color_
	        )
        )

	class HandlerWedge(HandlerPatch):
	    def create_artists(self, legend, orig_handle,
	                       xdescent, ydescent, width, height, fontsize, trans):
	        center = 0.3 * width - 0.1 * xdescent, 0.5 * height - 0.5 * ydescent
	        p = Wedge(center,width-xdescent,theta1=-20,theta2=20)
	        self.update_prop(p, orig_handle, legend)
	        p.set_transform(trans)
	        return [p]
