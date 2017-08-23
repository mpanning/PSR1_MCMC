C Output from Public domain Ratfor, version 1.0
c      subroutine ltable(ifin,iout,ioeig,mtype)
      subroutine ltable(mtype,epsarg,npow,dt,fnyquist,nbran,cmin,cmax,
     *  maxlyr,modearray,nmodes)
      implicit real*8(a-h,o-z)
      parameter (MAXMODES=1000)
      integer*4 ls0,m1,kind,null
      real*4 tlen,dcda,dcdb,dcdrh,y1,dy1dr,dum,vphase,gcom,qmod,fl4,sqll
     *,ekin, wdiff,w4,zero,y2,dy2dr,rn4,vp4,vs4,rh4,q4,df4,cmax4,dcdxi
      common/bits/pi,rn,vn,wn,w,wsq,wray,qinv,cg,tref,fct,eps,fl, fl1,fl
     *2,fl3,sfl3,nord,l,kount,knsw,ifanis,iback
      common/shanks/b(46),c(10),dx,step(8),stepf,maxo,in
      common/eifx/a(5,1500),dcda(1500),dcdb(1500),dcdrh(1500),dcdxi(1500
     *), y1(1500),dy1dr(1500),y2(1500),dy2dr(1500),dum(15*1500)
      common/stdep/ls,maxlayr
      common/rindx/nic,noc,nsl,nicp1,nocp1,nslp1,n
      dimension xl1(0:600),xl2(0:600),d1(0:600),d2(0:600),k1(0:600),k2(0
     *:600)
      dimension xlpred(0:600),dlpred(0:600)
      character*2 ichar(2)
      data nmx/600/,inss/5/,ichar/' R',' L'/,zero/0./,null/0/,epsxl/0.00
     *1/
      real*8 modearray(7,MAXMODES)
      integer*4 nmodes,mtype,npow,nbran,maxlyr
      real*8 epsarg,dt,fnyquist,cmin,cmax
Cf2py intent(in) mtype,epsarg,npow,dt,fnyquist,nbran,cmin,cmax,maxlyr
Cf2py intent(out) modearray,nmodes

      nmodes=0
      chz=2.0d0*pi
      stepf=1.d0
c F2PY pass these as arguments into function
c      read(ifin,*)
c      read(ifin,*) eps
      eps=epsarg
      eps1=eps
      eps2=eps
      modtot=0
      call steps(eps)
c      read(ifin,*)
c      read(ifin,*) npow
c      read(ifin,*)
c      read(ifin,*) dt
      ttlen=dt*(2**npow)
      tlen=ttlen
c      read(ifin,*)
c      read(ifin,*) fnyquist
      if(fnyquist.le.0.)then
      fnyquist=1.0d0/(2.0d0*dt)
      endif
c      read(ifin,*)
c      read(ifin,*) nbran
c      read(ifin,*)
c      read(ifin,*) cmin,cmax
c      read(ifin,*)
c      read(ifin,*) maxlayr
      df=1.0d0/ttlen
      nfreq=fnyquist/df
c      write(iout,80) cmin,cmax,df,nfreq
c80    format(/,'Phase velocity bounds:',2g12.4,' km/s',//, 'Frequency sp
c     *acing ',g12.6,' Hz,  number of fft bins:',i6)
c      write(iout,100) eps,eps1
c100   format(/,'Integration precision =',g12.4,'  root precision =', g12
c     *.4,///,3x,'mode',5x,'order', 7x,'f(hz)',10x,'t(secs)',4x,'phase ve
c     *l(km/s)',6x,'grp vel(km/s)', 8x,'q',13x,'raylquo',/)
      print *,'Frequency step=',df,' Hz'
      print *,'Highest frequency=',nfreq*df,' Hz'
      wmin=df*chz
      wmax=nfreq*wmin
      if(nbran.lt.0.or.nbran.gt.nmx)then
      nbran=nmx
      endif
c      open(10,file='dummyeig',form='unformatted')
c      read(10) rn4,in4,n4
      df4=df
      cmax4=cmax
c      rewind ioeig
c      write(ioeig) rn4,in4,n4,nfreq,nbran,df4,cmax4
c      do23004 i=1,n4 
c      read(10) rn4,vp4,vs4,rh4,q4
c      write(ioeig) rn4,vp4,vs4,rh4,q4
23004 continue
23005 continue
c      close(10)
      do23006 i=1,nfreq
      xlpred(i)=0.0
23006 continue
23007 continue
      do23008 i=1,nfreq
      dlpred(i)=0.0
23008 continue
23009 continue
      do23010 ifreq=1,nfreq 
      wrad=ifreq*wmin
      fhz=wrad/chz
c fix this for Mars
c      flmin=6371.0d0*wrad/cmax - 0.5d0
      flmin=rn*1.d-3*wrad/cmax - 0.5d0
      if(flmin.lt.10.0d0)then
      flmin=10.0d0
      endif
c fix this for Mars
c      flmax=6371.0d0*wrad/cmin - 0.5d0
      flmax=rn*1.d-3*wrad/cmin - 0.5d0
      if(flmax.lt.10.0d0)then
      flmax=10.0d0
      endif
      if(flmax.le.flmin)then
      goto 23010
      endif
      knsw=1
      maxo=inss
      call ldetqn(flmin,wrad,kmax,dmax,0,nerr)
      if(nerr.gt.0)then
      print *,'ERROR in ldetqn, frequency=',flmin
      kmax=mmax
      endif
      call ldetqn(flmax,wrad,kmin,dmin,0,nerr)
      if(nerr.gt.0)then
      print *,'ERROR in ldetqn, frequency=',flmax
      kmin=mmin
      endif
      mmax=kmax
      mmin=kmin+1
20    format(i6,f12.6,2i6)
      if(mmax.lt.mmin)then
      goto 23010
      endif
      if(mmax.gt.nbran)then
      mmax=nbran
      endif
      do23026 m=mmin,mmax 
      xl1(m)=flmin
      k1(m)=kmax
      d1(m)=dmax
      xl2(m)=flmax
      k2(m)=kmin
      d2(m)=dmin
23026 continue
23027 continue
      nloss=0
      do23028 m=mmin,mmax 
23030 if(k1(m).ne.k2(m)+1)then
      fltryold=fltry
      fltry=0.5d0*(xl1(m)+xl2(m))
      if(abs(fltryold-fltry).lt.epsxl .or. abs(xl1(m)-xl2(m)).lt.epsxl .
     *or. k1(m).lt.k2(m))then
      nloss=nloss+1
      write(13,*) 'Loss of precision, m=',m,', xl=',xl1(m),xl2(m)
      write(13,*) 'k1,k2=',k1(m),k2(m),' fltry=',fltry,' ifreq,f='
      write(13,20) ifreq,fhz,mmin,mmax
      if(nloss.gt.10)then
      stop 'too many loss of precision cases'
      endif
      xl2(m)=0.
      goto 123
      endif
      call ldetqn(fltry,wrad,ktry,dtry,0,nerr)
      if(nerr.gt.0)then
      print *,'WARNING: nerr=1 on ldetqn, ktry may be corrupt'
      endif
      do23038 mm=ktry+1,mmax 
      if(xl2(mm).gt.fltry)then
      xl2(mm)=fltry
      k2(mm)=ktry
      d2(mm)=dtry 
      endif
23038 continue
23039 continue
      do23042 mm=mmin,ktry 
      if(xl1(mm).lt.fltry)then
      xl1(mm)=fltry
      k1(mm)=ktry
      d1(mm)=dtry 
      endif
23042 continue
23043 continue
      goto 23030
      endif
23031 continue
123   continue
23028 continue
23029 continue
      nloss=0
      do23046 m=mmin,mmax 
      knsw=0
      maxo=8
      if(xl2(m).lt.10.0d0)then
      goto 23046
      endif
      call rootf(wrad,flroot,detroot,xl1(m),xl2(m),d1(m),d2(m),eps)
      if(flroot.lt.10.0d0)then
      goto 23046
      endif
      call detqn(wrad,kroot,droot,1,nerr)
      wdiff=(wrad-wray*wn)/wrad
      gcom=vn*cg/1000.d0
      qmod=0.0
      if(qinv .gt. 0.0)then
      qmod=1.0/qinv
      endif
c will need to fix this for Mars
c      vphase=6371.d0*wrad/(fl+0.5d0)
      vphase=rn*1.d-3*wrad/(fl+0.5d0)
      w4=wrad
      ekin=1.0
      fl4=fl
      sqll=sfl3
      ls00=max(ls,nsl-maxlayr)
      ls0=ls00
      m1=m+1
      kind=3-mtype
      if(nerr.gt.0)then
c write warning messages to stdout
c      write(iout,190) m,ichar(mtype),flroot,fhz,1.0/fhz,vphase,gcom,qmod
c     *,wdiff,'warn: detqn error'
      write(6,190) m,ichar(mtype),flroot,fhz,1.0/fhz,vphase,gcom,qmod
     *,wdiff,'warn: detqn error'
      endif
      if(abs(wdiff).gt.0.2)then
c      write(iout,190) m,ichar(mtype),flroot,fhz,1.0/fhz,vphase,gcom,qmod
c     *,wdiff,'skip: inaccurate'
      write(6,190) m,ichar(mtype),flroot,fhz,1.0/fhz,vphase,gcom,qmod
     *,wdiff,'skip: inaccurate'
190   format(i5,a2,f10.2,6g16.7,1x,a)
      goto 23046
      endif
      isig=+1
      if(y1(ls).lt.0.)then
      isig=-1
      do23060 i=ls00,n 
      y1(i)=-y1(i)
      dy1dr(i)=-dy1dr(i)
      if(kind.eq.1)then
      goto 23060
      endif
      y2(i)=-y2(i)
      dy2dr(i)=-dy2dr(i)
23060 continue
23061 continue
      endif
c write to modearray rather than file
c      write(iout,200) m,ichar(mtype),flroot,fhz,1.0/fhz,vphase,gcom,qmod
c     *,wdiff,isig
c200   format(i5,a2,f10.2,6g16.7,i5)
      nmodes=nmodes+1
      modearray(1,nmodes)=real(m)
      modearray(2,nmodes)=flroot
      modearray(3,nmodes)=fhz
      modearray(4,nmodes)=vphase
      modearray(5,nmodes)=gcom
      modearray(6,nmodes)=qmod
      modearray(7,nmodes)=wdiff
c Skip writing eigenfunctions
      gcom=gcom*1000.0
      vphase=vphase*1000.0
      if(qmod.gt.0.)then
      qmod=1.0/qmod
      endif
c      write(ioeig) w4,vphase,gcom,qmod,ekin,wdiff,fl4,sqll,ls0,m1,kind
c      if(kind.eq.1)then
c      do23068 i=ls00,n
c      write(ioeig) y1(i),dy1dr(i),dcda(i),dcdb(i),dcdrh(i),dcdxi(i)
c23068 continue
c23069 continue
c      else
c      do23070 i=ls00,n
c      write(ioeig) y1(i),dy1dr(i),y2(i),dy2dr(i),dcda(i),dcdb(i),dcdrh(i
c     *),dcdxi(i)
c23070 continue
c23071 continue
c      endif
23046 continue
23047 continue
23010 continue
23011 continue
c      write(ioeig) zero,zero,zero,zero,zero,zero,zero,zero,null,null,nul
c     *l
      return
      end
      subroutine ldetqn(fldum,wdum,kdum,ddum,ifeif,nerr)
      implicit real*8(a-h,o-z)
      real*4 f4,d4
      common/bits/pi,rn,vn,wn,w,wsq,wray,qinv,cg,tref,fct,eps,fl, fl1,fl
     *2,fl3,sfl3,nord,l,kount,knsw,ifanis,iback
      common/shanks/b(46),c(10),dx,step(8),stepf,maxo,in
      data nmx/600/,inss/5/
      l=fl
      fl=fldum
      fl1=fl+1.d0
      fl2=fl+fl1
      fl3=fl*fl1
      sfl3=dsqrt(fl3)
      call detqn(wdum,kdum,ddum,ifeif,nerr)
      f4=fl
      d4=ddum
      return
      end
      subroutine rootf(wrad,flroot,detroot,x1,x2,d1,d2,tol)
      implicit real*8 (a-h,o-z)
      parameter (itmax=100,eps=3.d-8)
      a=x1
      b=x2
      fa=d1
      fb=d2
      if(fb*fa.gt.0.)then
      stop 'root must be bracketed for rootf.'
      endif
      fc=fb
      do23074 iter=1,itmax 
      if(fb*fc.gt.0.)then
      c=a
      fc=fa
      d=b-a
      e=d
      endif
      if(abs(fc).lt.abs(fb))then
      a=b
      b=c
      c=a
      fa=fb
      fb=fc
      fc=fa
      endif
      tol1=2.*eps*abs(b)+0.5*tol
      xm=.5*(c-b)
      if(abs(xm).le.tol1 .or. fb.eq.0.)then
      flroot=b
      return
      endif
      if(abs(e).ge.tol1 .and. abs(fa).gt.abs(fb))then
      s=fb/fa
      if(a.eq.c)then
      p=2.*xm*s
      q=1.-s
      else
      q=fa/fc
      r=fb/fc
      p=s*(2.*xm*q*(q-r)-(b-a)*(r-1.))
      q=(q-1.)*(r-1.)*(s-1.)
      endif
      if(p.gt.0.)then
      q=-q
      endif
      p=abs(p)
      if(2.*p .lt. min(3.*xm*q-abs(tol1*q),abs(e*q)))then
      e=d
      d=p/q
      else
      d=xm
      e=d
      endif
      else
      d=xm
      e=d
      endif
      a=b
      fa=fb
      if(abs(d) .gt. tol1)then
      b=b+d
      else
      b=b+sign(tol1,xm)
      endif
      call ldetqn(b,wrad,kroot,detroot,0,nerr)
      fb=detroot
23074 continue
23075 continue
      print *, 'rootf exceeding maximum iterations.'
      flroot=b
      return
      end
      subroutine partials(y,dw,i)
      implicit real*8 (a-h,o-z)
      real*8 lcon,ncon,lspl,nspl
      common/bits/pi,rn,vn,wn,w,wsq,wray,qinv,cg,tref,fct,eps,fl, fl1,fl
     *2,fl3,sfl3,nord,ll,kount,knsw,ifanis,iback
      common r(1500),fmu(1500),flam(1500),qshear(1500),qkappa(1500), xa2
     *(1500),xlam(1500),rho(1500),qro(3,1500),g(1500),qg(3,1500), fcon(1
     *500),fspl(3,1500),lcon(1500),lspl(3,1500),ncon(1500), nspl(3,1500)
     *,ccon(1500),cspl(3,1500),acon(1500),aspl(3,1500)
      dimension y(4),dw(4)
      a=acon(i)
      f=fcon(i)
      c=ccon(i)
      xl=lcon(i)
      xn=ncon(i)
      d=rho(i)
      z=r(i)
      z2=z*z
      y1=y(1)
      y3=y(3)
      y1r=y1/z
      y3r=y3/z
      t1=2.0d0*y1-fl3*y3
      y2=c*y(2)+f*t1/z
      y4=xl*(y(4)+y1r-y3r)
      vsv=xl/d
      if(xl.gt.0.0d0)then
      vsv=dsqrt(vsv)
      endif
      vph=sqrt(a/d)
      t2=t1*t1
      t3=a-xl-xl
      dw(3)=-wsq*d*z2*(y1*y1+fl3*y3*y3)+z2*y2*y2/c+ (a-xn-f*f/c)*t2+(fl-
     *1.0d0)*fl3*(fl+2.0d0)*xn*y3*y3
      if(xl.gt.0.0d0)then
      dw(3)= fl3*z2*y4*y4/xl+dw(3)
      endif
      dw(3)=dw(3)-2.0d0*d*z*y1*g(i)*t1
      dw(3)=0.5d0*w*dw(3)/d
      dw(1)=((z*y2+2.0d0*f*xl*t1/t3)**2)/c+a*t2*(1.0d0-f*f*a/(c*t3*t3))
      dw(1)=w*dw(1)/vph
      if(xl.gt.0.0d0)then
      dw(2)=fl3*z2*y4*y4/xl-4.0d0*f*xl*z*y(2)*t1/t3+2.0d0*xn*(fl3*y3*(y1
     *- y3)-y1*t1)
      dw(2)=w*dw(2)/vsv
      else
      dw(2)=0.0d0
      endif
      dw(4)=w*xn*(fl3*y3*(y1-y3)-y1*t1)
      return
      end
