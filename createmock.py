import scipy
import numpy as np
#import halocorr_with_alpha as h
#import haloproperty_mass as hcm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#import halo_shape as hs
import time
class createmock(object):
	def __init__(self,mvir,m200b,m200c,alpha,haloprops,k,Tfn,alphaprop=["binned",10],Omega_matter = 0.276,Omega_lambda = 0.724,H_0=70.,ns=0.961, sigma_8 = 0.811 ,Omega_baryon = 0.045,z=0):
#		hcm.haloproperty_mass.__init__(self,mvir=mvir,m200b=m200b,m200c=m200c,k=k,Tfn=Tfn,Omega_matter = Omega_matter,Omega_lambda = Omega_lambda ,H_0=H_0,ns=ns ,sigma_8 = sigma_8 ,Omega_baryon = Omega_baryon)
#		h.halocorr_with_alpha.__init__(self,z=z)
		self.Nhalo = len(m200b)
		self.haloprops = haloprops
		self.m200b = m200b  ## M200b in Msun h^{-1}
		self.m200c = m200c
		self.alpha = alpha
		self.returnval = alphaprop[0]
		self.alphabins = alphaprop[1]
		self.k = k ### k in h Mpc^1
		self.Tfn =Tfn
		self.z =z
		
	def _muSigma(self):		
		"""
		returns an array of mean-alpha and std-alpha values for every input alpha value
		"""
		meanalpha,edges,binno=scipy.stats.binned_statistic(np.log(np.log(self.m200b)), np.log(self.alpha), statistic='mean', bins=self.alphabins)	
		plt.plot(1/2.*(edges[1:]+edges[0:-1]),meanalpha)
		plt.savefig('alphainterpolatetest.png')
		print ('meanVedges',meanalpha,edges)
		y = np.log(self.alpha)
		scatteralpha,edges,binno = scipy.stats.binned_statistic(np.log(self.m200b), y, statistic=lambda y: (np.percentile(y,84.15)-np.percentile(y,15.85))/2, bins=self.alphabins)
		if self.returnval=='binned':
			mean = meanalpha[binno-1]
			std = scatteralpha[binno-1]
		elif self.returnval == 'interpolate':
			fmu = interp1d(1/2.*(edges[1:]+edges[0:-1]),meanalpha, kind='cubic',fill_value="extrapolate")
			fstd = interp1d(1/2.*(edges[1:]+edges[0:-1]),scatteralpha, kind='cubic',fill_value="extrapolate")
			mean = fmu(np.log(self.m200b))
			std = fstd(np.log(self.m200b))
			print ('needs to be filled currently cubic spline')
		elif self.returnval == 'fittingfn':
			print ('needs to be filled')
		return mean,std

	@property
	def alpha_tilde(self):
		"""
		input:
		alpha - array of property alpha
		returns standardised variable
		"""
		mean,sigma = self._muSigma()
		return (np.log(self.alpha)-mean)/sigma

	@property
	def nu_peak(self):
		"""
		returns peakheight for all the input haloes computed using m200b
		"""
		return self.PeakHeight(self.m200b,self.k,self.Tfn,self.z)


	@staticmethod
	def get_lognormal(c,avg_logc,std_logc):
		"""
		input:
		c - standard gaussian variable c = (lnC- <lnC>)/Std(lnC)
		avg_logc - <lnC>
		std_logc - Std(lnC)
		returns the lognormal value C
		"""
		return np.exp(c*std_logc+avg_logc)  

	def makemock(self,haloprop,z=0):
		rhoget = getattr(self,'rho_'+haloprop[0])
		rho = rhoget(self.nu_peak)
		print ('rhomax',rho.max(),'rhomin',rho.min())
		print ('alphamaxmin',self.alpha.max(),self.alpha.min())
		begin = time.time()
		c = self.sample_conditionalP(self.alpha_tilde,rho)
		print (time.time()-begin)
		print ('c',c)
		if len(haloprop)==1:
			return c
		if haloprop[0] in ['b_to_a','beta','triaxiality','vc_to_va']:
			print ("needs tddo be filled")
		elif haloprop[0] in ['Spin','spin_bullock','c200b','c200c','c_to_a']:
			avg_logc = getattr(self,"ln"+haloprop[0]+"_"+haloprop[1])
			std_logc = getattr(self,"sigln"+haloprop[0]+"_"+haloprop[2])
			C = self.get_lognormal(c,avg_logc(),std_logc())
		return C

	def sample_conditionalP(self,alphat,rho):
		"""
		samples the conditional probability distribution p(c|\alpha,m) = N(rhoalpha,(1-rho^2))
		"""
		return np.random.normal(loc=rho*alphat,scale=np.sqrt(1-rho**2))
  
