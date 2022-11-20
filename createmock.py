import scipy
import numpy as np
#import halocorr_with_alpha as h
#import haloproperty_mass as hcm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy import stats
import os
absolutepathcab=  os.path.dirname(os.path.abspath(__file__))

#import halo_shape as hs
import time
class createmock(object):
	def __init__(self,m200b,m200c,alpha,k,Tfn,alphaprop=["binned",10],Omega_matter = 0.276,Omega_lambda = 0.724,H_0=70.,ns=0.961, sigma_8 = 0.811 ,Omega_baryon = 0.045,z=0):
#		hcm.haloproperty_mass.__init__(self,mvir=mvir,m200b=m200b,m200c=m200c,k=k,Tfn=Tfn,Omega_matter = Omega_matter,Omega_lambda = Omega_lambda ,H_0=H_0,ns=ns ,sigma_8 = sigma_8 ,Omega_baryon = Omega_baryon)
#		h.halocorr_with_alpha.__init__(self,z=z)
		self.Omega_matter = Omega_matter
		self.ns = ns
		self.sigma_8 = sigma_8
		self.Nhalo = len(m200b)
		self.m200b = m200b  ## M200b in Msun h^{-1}
		self.m200c = m200c
		self.alpha = alpha
		self.returnval = alphaprop[0]
		self.alphabins = alphaprop[1]
		self.k = k ### k in h Mpc^1
		self.Tfn =Tfn
		self.z =z
		self.H_0 = H_0
		self.rho_c_h2_msun_mpc3 = 2776*1e8    ## in (msun/h)(mpc/h)**3 is the critical density today (ie redshift evolution not included)
		self.Omega_lambda = Omega_lambda
		self.delta_c=1.686
		
		

		
	def _muSigma(self):		
		"""
		returns an array of mean-alpha and std-alpha values for every input alpha value
		"""
		meanalpha,edges,binno=scipy.stats.binned_statistic((np.log(self.m200b)), np.log(self.alpha), statistic='mean', bins=self.alphabins)	
#		plt.plot(1/2.*(edges[1:]+edges[0:-1]),meanalpha)
#		plt.savefig('alphainterpolatetest.png')
		y = np.log(self.alpha)
		scatteralpha,edges,binno = scipy.stats.binned_statistic(np.log(self.m200b), y, statistic=lambda y: (np.percentile(y,84.15)-np.percentile(y,15.85))/2, bins=self.alphabins)
		if self.returnval=='binned':
			mean = meanalpha[binno-1]
			std = scatteralpha[binno-1]
		elif self.returnval == 'interpolate':
			fmu = interp1d(1/2.*(edges[1:]+edges[0:-1]),meanalpha, kind='linear',fill_value="extrapolate")
			fstd = interp1d(1/2.*(edges[1:]+edges[0:-1]),scatteralpha, kind='linear',fill_value="extrapolate")
			mean = fmu((np.log(self.m200b)))
			std = fstd(np.log(self.m200b))
#			plt.scatter((np.log(self.m200b)),mean,c='k')
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

	def makemock(self,haloprop):
#		rhoget = getattr(self,'rho_'+haloprop)
		rho = self.rho_hprop(haloprop,self.nu_peak)
		begin = time.time()
		C= self.sample_conditionalP(self.alpha_tilde,rho)
		if haloprop in ['Spin']:
			muln = self.mu_hprop(haloprop,self.nu_peak)
			sigln = self.sig_hprop(haloprop,self.nu_peak) 
			C = self.get_lognormal(C,muln,sigln)
		elif haloprop in ["c_to_a","vc_to_va"]:
			mu = self.mu_hprop(haloprop,self.nu_peak)
			sig = self.sig_hprop(haloprop,self.nu_peak) 
			C = C*sig + mu
		elif haloprop in ["beta"]:
			mu = self.mu_hprop(haloprop,self.nu_peak)
			sig = self.sig_hprop(haloprop,self.nu_peak)
			C = 1-np.exp(C*sig + mu)
		elif haloprop in ["c200b"]:
			muln = self.lnc200b_DK15(self,z=self.z,m200c=[],mflag=0)
			sigln = self.siglnc200b_DK15()
			C = self.get_lognormal(C,muln,sigln)
#		if len(haloprop)==1:
#			return c
#		if haloprop[0] in ['b_to_a','beta','triaxiality','vc_to_va']:
#			print ("needs tddo be filled")
#		elif haloprop[0] in ['Spin','c200b']:
#			avg_logc = getattr(self,"ln"+haloprop[0]+"_"+haloprop[1])
#			std_logc = getattr(self,"sigln"+haloprop[0]+"_"+haloprop[2])
#			C = self.get_lognormal(c,avg_logc(),std_logc())
		return C

	def sample_conditionalP(self,alphat,rho):
		"""
		samples the conditional probability distribution p(c|\alpha,m) = N(rhoalpha,(1-rho^2))
		"""
		return np.random.normal(loc=rho*alphat,scale=np.sqrt(1-rho**2))
  
	def rho_hprop(self,hprop,v,sample_cov=0,sampling=0):
		"""
		v is the peakheight		
		"""
		if sample_cov ==0:
			roload = np.load(absolutepathcab+"/fits/pearson_rho_"+hprop+"-alpha.npz")
			rhofit = roload['name1']
			rho = np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(roload['name1'],self.roload['name2'],sampling)
			rho= 0	
			for i in range(len(roload['name1'])):
				rho +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		return rho
		
	def mu_hprop(self,hprop,v,sample_cov=0,sampling=0):
		"""
		v is the peakheight		
		"""
		if sample_cov ==0:
			load = np.load(absolutepathcab+"/fits/pearson_mu_"+hprop+"-alpha.npz")
			rhofit = load['name1']
			mu = np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(load['name1'],self.load['name2'],sampling)
			mu= 0	
			for i in range(len(load['name1'])):
				mu +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		return mu

	def sig_hprop(self,hprop,v,sample_cov=0,sampling=0):
		"""
		v is the peakheight		
		"""
		if sample_cov ==0:
			load = np.load(absolutepathcab+"/fits/pearson_sig_"+hprop+"-alpha.npz")
			rhofit = load['name1']
			sig= np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(load['name1'],self.load['name2'],sampling)
			sig= 0	
			for i in range(len(load['name1'])):
				sig +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		return sig


	def PeakHeight(self,mass,k,Tfn,z):
		"""
		Inputs:
		mass in Msunh-1
		k in h Mpc^1
		z redshift
		T the tranfer function
		"""
		R = (3/(4*np.pi)*mass/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.) ## is in units of Mpch-1
		PS = self.PS(k,z,Tfn)
		sigma_square = np.zeros([len(R),1])
		for i in range(0,len(R)):
			wk = self.Wk(k,R[i])
			sigma_square[i] = 1/(2.*np.pi**2)*np.trapz(PS*wk**2*k**2,k)
		nu = self.delta_c/np.sqrt(sigma_square)
		return nu.flatten()
		
	def PS(self,k,z,T):
		"""
		Input
		k in h Mpc^1
		z redshift
		T the tranfer function
		
		Outputs 
		Pk the power spectrum
		"""
		R8=8.
		integrand = 1/(2*np.pi**2)*k**(self.ns+2.)*T**2*self.Wk(k,R8)**2
		igrate = np.trapz(integrand,k)
		SigmaSquare=self.sigma_8**2
		NormConst = SigmaSquare/igrate
		return NormConst*k**self.ns*(T)**2*self.GrowthFunctionAnalytic(1./(1.+z))**2/self.GrowthFunctionAnalytic(1)**2

	def GrowthFunctionAnalytic(self,a):
		"""
		check why is H defined as a series on ones????
		"""
		a=np.array(a)+1e-15
		D=np.ones(np.size(a))
		H=np.ones(np.size(a))
		H=self.H_0*(self.Omega_matter*(a**(-3))+self.Omega_lambda)**(1/2.)
		D=(H/self.H_0)*a**(5/2.)/np.sqrt(self.Omega_matter)*sp.hyp2f1(5/6.,3/2.,11/6.,-a**3*self.Omega_lambda/self.Omega_matter)
		return D

	def Wk(self,k,R):
		"""
		Fourier Transform of a Spherical Top Hat Filter
		"""
		return 3/(k*R)**3*(np.sin(k*R)-(k*R)*np.cos(k*R))

	def lnc200b_DK15(self,z=0,m200c=[],mflag=0):
		"""
		Inputs m200c?
		"""
		if len(m200c)==0:
			c200c = self.cmz_CDM_DK15(self.m200c,z)
		else:
			c200c = self.cmz_CDM_DK15(m200c,z)
		Deltaref = 200*self.E(z)**2/(self.Omega_matter*(1+z)**3)
		print ('deltaref',Deltaref)
		Delta = 200
		if mflag==0:
			return np.log(self.cDelta(c200c,Deltaref,Delta)) 
		elif mflag==1:
			c200b = (self.cDelta(c200c,Deltaref,Delta)) 
			m200b = self.MDelta(m200c,c200b,Deltaref,Delta)
			return np.log(c200b),m200b

	def cDelta(self,cref,Deltaref,Delta):
		""" Convert from c_ref to c_Delta using HK03 prescription. Delta is what multiplies rho_b (not rho_crit)."""
		fDelta = Delta/Deltaref*self.f_nfw(cref)
		out = self.c_hk03(fDelta)
		return out
    # ~ ############################################################
    # ~ ############################################################
	def MDelta(self,Mref,cref,Deltaref,Delta):
		""" Convert from M_ref,c_ref to M_Delta using HK03 prescription. Delta is what multiplies rho_b (not rho_crit)."""
		out = (self.cDelta(cref,Deltaref,Delta)/cref)**3
		out *= (Mref*(Delta/Deltaref))
		return out

	def cmz_CDM_DK15(self,m,z):
		""" Fitting function c(nu(m),z) from Diemer&Kravtsov15 arXiv:1407.4730.
		"""
		nspec = self.calc_nspec_DK15(m)
		nu = self.PeakHeight(m,self.k,self.Tfn,z)	
		out = self.cnuz_CDM_DK15(nu,z,nspec)
		return out

	def siglnc200b_DK15(self):
		sigma_lnc = np.ones(np.size(self.m200b))*0.16*np.log(10)
		return sigma_lnc
		
	def E(self,z,ok=0):
		"""
		seems like H/H_0 need to confirm
		"""
		a = 1./(1.+z)
		# ~ return self.H_0*(self.Omega_matter*a**(-3) + self.Omega_lambda + ok*(a)**(-2)+(1-self.Omega_lambda-self.Omega_matter)*a**(-4))**(1/2)
		if ok==0:
			# ~ return self.H_0*(self.Omega_matter*a**(-3) + (1-self.Omega_matter-9.23640e-5)+(9.23640e-5)*a**(-4))**(1/2)
			return (self.Omega_matter*a**(-3) + (self.Omega_lambda))**(1/2)
		else:
			return (self.Omega_matter*a**(-3) + (1-self.Omega_matter-ok-8e-5) + ok*(a)**(-2)+(8e-5)*a**(-4))**(1/2)

