#Argon
#Age_inverse
#Dt_withpressure
#Alpha_make
#Matrix_plane;Matrix_sphere;Matrix_cylinder
#Age_frac;Heat_frac



import csv
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches

plt.rc('font', family='serif') 
plt.rc('font', serif='Times New Roman') 



def Argon(p_total40,p_total39,JJ):
        #print(np.shape(p_total40),np.shape(p_total39))
        #print(type(p_total40),type(p_total39))
   
        age_million=np.multiply(np.divide(1,Lamdba_age),np.log(np.add(np.multiply(JJ,np.divide(p_total40,p_total39)),1)))
        #print('million ',age_million)
        return(age_million)

def Age_inverse(age):
    
        age=age*1
        F=(math.exp(Lamdba_age*age)-1)/JJ
        return(F)

def Dt_age_withpressure(p_temprature,D_u,Ea,p_Press,p_volume):
        D_u=D_u*1000000*12*30*24*60*60    
        p_Dt=np.multiply(D_u,np.exp(np.divide(-np.add(Ea,np.multiply(p_Press,p_volume)),np.multiply(R,np.add(p_temprature,273)))))
        return(p_Dt)

def Dt_heat_withpressure(p_temprature,D_u,Ea,p_Press,p_volume):  
        p_Dt=np.multiply(D_u,np.exp(np.divide(-np.add(Ea,np.multiply(p_Press,p_volume)),np.multiply(R,np.add(p_temprature,273)))))
        return(p_Dt)

def Alpha_make(p_Dt,dt,dx):
    
        alpha=np.divide(np.multiply(p_Dt,dt),np.multiply(np.power(dx,2),1))
        return(alpha)

def Grid_make(p_Node_space,p_Length_space,p_Length_time,p_Node_time):

        dx = float(p_Length_space)/float(p_Node_space-1)
        x_grid = np.array([j*dx for j in range(p_Node_space)])

        dt = np.array([float(p_Length_time)/float(p_Node_time-1) for i in range(p_Node_time)])
        t_grid = np.array([n*dt for n in range(p_Node_time)])
        print(len(t_grid))
              
        return(dt,dx,x_grid,t_grid)

def generateRHS(T, r):

        b = T[1:-1]*2*(1-r) + r*T[:-2] + r*T[2:]
        print('b in0,' ,len(b))
        b=list(b)
        print('b in,' ,len(b))
        b.append(T[-1]*2*(1-r)+2*r*T[-2])
        b=np.array(b)

        b[0] = T[0]
        #T[-1]+=b[-1]

        return b

def generateMatrix(J, r):

        d = np.diag(np.ones(J-1)*2*(1+r))
        d[-1,-1] = r+2
        ud = np.diag(np.ones(J-2)*r*(-1), 1)
        ld = np.diag(np.ones(J-2)*r*(-1), -1)    
        A = d + ud + ld
        
        return A


def Matrix_creat_plane(p_alpha,p_Node_space,p_Length_space,p_Length_time,p_Node_time):

    
    A_u=np.diagflat([-p_alpha]+[-p_alpha for i in range(p_Node_space-2)],1)\
        +np.diagflat([0.0*(1+p_alpha)*2]+[(2+2*p_alpha) for i in range(p_Node_space-1)])\
        +np.diagflat([-p_alpha]+[-p_alpha for i in range(p_Node_space-3)]+[-2*p_alpha],-1)

    B_u=np.diagflat([p_alpha]+[p_alpha for i in range(p_Node_space-2)],1)\
        +np.diagflat([0.0*(1-p_alpha)*2]+[(2-2*p_alpha) for i in range(p_Node_space-1)])\
        +np.diagflat([p_alpha]+[p_alpha for i in range(p_Node_space-3)]+[2*p_alpha],-1)

    C_u=np.diagflat([(p_alpha)]+[-p_alpha for i in range(p_Node_space-2)],1)\
        +np.diagflat([p_alpha]+[(1+2*p_alpha) for i in range(p_Node_space-1)])\
        +np.diagflat([-p_alpha for i in range(p_Node_space-2)]+[-2*p_alpha],-1)
    
    return(A_u,B_u,C_u)

def Matrix_creat_sphere(p_alpha,p_Node_space,p_Length_space,p_Length_time,p_Node_time):

    index=2
    A_u=np.diagflat([(-p_alpha)]+[-p_alpha*(1-1/max((i-1)*index,1)) for i in range(p_Node_space-2)],1)\
        +np.diagflat([2*(1+index)*p_alpha]+[(1+2*p_alpha) for i in range(p_Node_space-1)])\
        +np.diagflat([-p_alpha*(1+1/max((i-1)*index,2)) for i in range(p_Node_space-2)]+[-2*p_alpha],-1)

    B_u=np.diagflat([(p_alpha)]+[p_alpha*(1-1/max((i-1)*index,1)) for i in range(p_Node_space-2)],1)\
        +np.diagflat([2*(1+index)*p_alpha]+[(1-2*p_alpha) for i in range(p_Node_space-1)])\
        +np.diagflat([p_alpha*(1+1/max((i-1)*index,2)) for i in range(p_Node_space-2)]+[2*p_alpha],-1)

    C_u=np.diagflat([(1+index)*(p_alpha)]+[-p_alpha*(1-1/max((p_Node_space-1-i)*index,1)) for i in range(p_Node_space-2)],1)\
        +np.diagflat([2*(1+index)*p_alpha]+[(1+2*p_alpha) for i in range(p_Node_space-1)])\
        +np.diagflat([-p_alpha*(1+1/max((p_Node_space-1-i)*index,2)) for i in range(p_Node_space-2)]+[-2*p_alpha],-1)
    

    return(A_u,B_u,C_u)


def Matrix_creat_cylinder(p_alpha,p_Node_space,p_Length_space,p_Length_time,p_Node_time):

    index=1
    A_u=np.diagflat([2*(1+index)*(p_alpha)]+[-p_alpha*(1-1/max((p_Node_space-1-i)*index,1)) for i in range(p_Node_space-2)],1)\
        +np.diagflat([2*(1+index)*p_alpha]+[(2+2*p_alpha) for i in range(p_Node_space-1)])\
        +np.diagflat([-p_alpha*(1+1/max((p_Node_space-1-i)*index,2)) for i in range(p_Node_space-2)]+[-2*p_alpha],-1)

    B_u=np.diagflat([-2*(1+index)*(p_alpha)]+[p_alpha*(1-1/max((p_Node_space-1-i)*index,1)) for i in range(p_Node_space-2)],1)\
        +np.diagflat([2*(1+index)*p_alpha]+[(2-2*p_alpha) for i in range(p_Node_space-1)])\
        +np.diagflat([p_alpha*(1+1/max((p_Node_space-1-i)*index,2)) for i in range(p_Node_space-2)]+[2*p_alpha],-1)

    C_u=np.diagflat([(1+index)*(p_alpha)]+[-p_alpha*(1-1/max((p_Node_space-1-i)*index,1)) for i in range(p_Node_space-2)],1)\
        +np.diagflat([2*(1+index)*p_alpha]+[(1+2*p_alpha) for i in range(p_Node_space-1)])\
        +np.diagflat([-p_alpha*(1+1/max((p_Node_space-1-i)*index,2)) for i in range(p_Node_space-2)]+[-2*p_alpha],-1)
    


    return(A_u,B_u,C_u)


def Matrix_creat_test(p_alpha,p_Node_space,p_Length_space,p_Length_time,p_Node_time):
    #sigma_u sholud be the mumber of the list type sigma_u
    
    A_u = np.diagflat([-p_alpha for i in range(p_Node_space-1)], -1) +\
          np.diagflat([1]+[1.+2.*p_alpha for i in range(p_Node_space-2)]+[1.+p_alpha]) +\
          np.diagflat([-p_alpha for i in range(p_Node_space-1)], 1)
        
    B_u = np.diagflat([p_alpha for i in range(p_Node_space-1)], -1) +\
          np.diagflat([1.-p_alpha]+[1.-2.*p_alpha for i in range(p_Node_space-2)]+[1.-p_alpha]) +\
          np.diagflat([p_alpha for i in range(p_Node_space-1)], 1)

    A_u[0][0]=0.0   
    B_u[0][0]=0.0

    return(A_u,B_u)

def Age_calcu_age(Temp_age,alpha_age,Node_space,Length_space,Length_age,Node_age,dt_age):
    Argon_record40=[]
    Fract_record40=[]
    Pressure=0
    Volume=0

    U=np.array([0.0 for i in range(Node_space)])
    for ti in range(0,Node_age):
        A_u,B_u,C_u=Matrix_creat_plane(alpha_age[ti],Node_space,Length_space,Length_age,Node_age)
        vect_C=lambda U:np.subtract(np.multiply(np.subtract(np.power(math.e,np.multiply(Lamdba_age,dt_age[ti])),1),ke),U)

        U_new = np.linalg.solve(A_u,B_u.dot(U)+vect_C(0))
        U=U_new
        U=[0.0 if i<0.0 else i for i in U]

        Argon_record40.append(U)
        Fract_record40.append(sum(U))


    return(Argon_record40,Fract_record40)

def Age_calcu_heat(U,alpha_heat,Node_space,Length_space,Length_heat,Node_heat):
    Argon_record=[]
    Fract_record=[]
    
    for ti in range(0,Node_heat):
        #alpha_heat=Alpha_make(Dt_heat,dt,dx)
        A_u,B_u,C_u=Matrix_creat_cylinder(alpha_heat[ti],Node_space,Length_space,Length_heat,Node_heat)
        U_new = np.linalg.solve(C_u,U)
        sub=np.subtract(U,U_new)
        U=U_new
        U=[0.0 if i<0.0 else i for i in U]
        Argon_record.append(U)
        Fract_record.append(sum(sub))

    return(Argon_record,Fract_record)


    

ke=0.1048

D0_p=10**(2.22)
Ea_p=37.037
D0_s=10**(1.97)
Ea_s=40.389
D0_c=10**(2.13)
Ea_c=39.477


JJ=0.004107
#decay frequence the unit should be -a this is ma
Lamdba_age=5.543*10**(-4)
#F=4.1987
R=1.987*10**(-3)

fuck_fraction=np.array([10.48,18.65,23.63,29.47,36.96,45.21,55.2,65.82,82.19,93.05,97.28,100])
fuck_age=np.array([245.2,225.1,219.4,219.1,219.6,217.7,224.8,237.2,252.6,273.3,279.4,275.8])
fuck_sigma=np.array([2.3,2.1,2.1,2.1,2.1,2.1,2.1,2.2,2.4,2.5,2.6,2.6])
fuck_fraction=np.multiply(fuck_fraction,0.01)





Node_space=100
Length_space=1
Length_age=400
Node_age=40
Length_heat=12*30*60
Node_heat=12
def dataset1():
        Temp_age=np.array([100 for i in range(Node_age)])
        Pressure_age=np.array([5 for i in range(Node_age)])
        Temp_heat=np.array([800,850,900,950,1000,1050,1100,1140,1180,1220,1300,1400])
        return Temp_age,Pressure_age,Temp_heat


def dataset2():
        Temp_age=np.array([150 for i in range(Node_age)])
        Pressure_age=np.array([5 for i in range(Node_age)])
        Temp_heat=np.array([800,850,900,950,1000,1050,1100,1140,1180,1220,1300,1400])
        return Temp_age,Pressure_age,Temp_heat

def dataset3():
        Temp_age=np.array([0 for i in range(21)]+[200]*3+[0 for i in range(24,Node_age)])
        Pressure_age=np.array([0 for i in range(Node_age)])
        Temp_heat=np.array([800,850,900,950,1000,1050,1100,1140,1180,1220,1300,1400])
        return Temp_age,Pressure_age,Temp_heat

def dataset4():
        Temp_age=np.array([0 for i in range(21)]+[190]*3+[0 for i in range(24,Node_age)])
        Pressure_age=np.array([0 for i in range(Node_age)])
        Temp_heat=np.array([800,850,900,950,1000,1050,1100,1140,1180,1220,1300,1400])
        return Temp_age,Pressure_age,Temp_heat

def dataset5():
        Temp_age=np.array([0 for i in range(21)]+[180]*3+[0 for i in range(24,Node_age)])
        Pressure_age=np.array([0 for i in range(Node_age)])
        Temp_heat=np.array([800,850,900,950,1000,1050,1100,1140,1180,1220,1300,1400])
        return Temp_age,Pressure_age,Temp_heat

def dataset6():
        Temp_age=np.array([0 for i in range(21)]+[170]*3+[0 for i in range(24,Node_age)])
        Pressure_age=np.array([0 for i in range(Node_age)])
        Temp_heat=np.array([800,850,900,950,1000,1050,1100,1140,1180,1220,1300,1400])
        return Temp_age,Pressure_age,Temp_heat

def dataset7():
        Temp_age=np.array([0 for i in range(21)]+[160]*3+[0 for i in range(24,Node_age)])
        Pressure_age=np.array([0 for i in range(Node_age)])
        Temp_heat=np.array([800,850,900,950,1000,1050,1100,1140,1180,1220,1300,1400])
        return Temp_age,Pressure_age,Temp_heat

def dataset8():
        Temp_age=np.array([0 for i in range(21)]+[150]*3+[0 for i in range(24,Node_age)])
        Pressure_age=np.array([0 for i in range(Node_age)])
        Temp_heat=np.array([800,850,900,950,1000,1050,1100,1140,1180,1220,1300,1400])
        return Temp_age,Pressure_age,Temp_heat

def dataset9():
        Temp_age=np.array([0 for i in range(21)]+[100]*3+[0 for i in range(24,Node_age)])
        Pressure_age=np.array([0 for i in range(Node_age)])
        Temp_heat=np.array([800,850,900,950,1000,1050,1100,1140,1180,1220,1300,1400])
        return Temp_age,Pressure_age,Temp_heat

        
#Temp_age,Pressure_age,Temp_heat=dataset1()        

def runmode(mode,Temp_age,Pressure_age,Temp_heat):
        if mode==1:
                Ea=Ea_p
                D0=D0_p
        elif mode==2:
                Ea=Ea_s
                D0=D0_s                
        elif mode==3:
                Ea=Ea_c
                D0=D0_c  
                
        mode_ration=Age_inverse(400)
        print('mode_ration is',mode_ration)
        constant_age=np.array([0 for i in range(Node_age)])

        Dt_heat=Dt_heat_withpressure(Temp_heat,D0,Ea,0,0)
        Dt_age=Dt_age_withpressure(Temp_age,D0,Ea,0,0)
        Dt_constant=Dt_age_withpressure(constant_age,D0,Ea,0,0)

        dt_heat,dx_heat,x_grid_heat,t_grid_heat=Grid_make(Node_space,Length_space,Length_heat,Node_heat)
        dt_age,dx_age,x_grid_age,t_grid_age=Grid_make(Node_space,Length_space,Length_age,Node_age)

        alpha_heat=Alpha_make(Dt_heat,dt_heat,dx_heat)
        alpha_age=Alpha_make(Dt_age,dt_age,dx_age)#should be temperature???
        alpha_constant=Alpha_make(Dt_constant,dt_age,dx_age)
        
        Argon_record_age_constant,Fract_record_age=Age_calcu_age(constant_age,Dt_constant,Node_space,Length_space,Length_age,Node_age,dt_age)
        Argon_record_age_40,Fract_record_age_40=Age_calcu_age(Temp_age,alpha_age,Node_space,Length_space,Length_age,Node_age,dt_age)
        

        Argon_mode_age_40=Argon_record_age_40[Node_age-1]
        Argon_mode_age_constant=Argon_record_age_constant[Node_age-1]

        Argon_mode_age_39=np.array([mode_ration for i in range(Node_space)])
        Argon_mode_age_39=np.divide(Argon_mode_age_constant,Argon_mode_age_39)

        Argon_record_heat_40,Fract_record_heat_40=Age_calcu_heat(Argon_record_age_40[Node_age-1],alpha_heat,Node_space,Length_space,Length_heat,Node_heat)
        Argon_record_heat_39,Fract_record_heat_39=Age_calcu_heat(Argon_mode_age_39,alpha_heat,Node_space,Length_space,Length_heat,Node_heat)
        age_mode=Argon(Fract_record_heat_40,Fract_record_heat_39,JJ)
        total39=[sum(Fract_record_heat_39[:i])/sum(Fract_record_heat_39) for i in range(len(Fract_record_heat_39))]
        print(Fract_record_heat_40)
        print(Fract_record_heat_39)
        print(age_mode)
        print('total39 is ',total39)


        #plot modeual
        fig,axes=plt.subplots(figsize=(10,16),nrows=2,ncols=3)

        ##subplot1    
        plot1=plt.subplot(311)
        plot1_1=plot1.twinx()

        plot1.plot(t_grid_age,Temp_age)
        plot1.set_ylim(0,600)
        plot1_1.plot(t_grid_age,Pressure_age,'b')
        plot1_1.set_ylim(0,6)
        plot1.set_xlabel('Age(Ma)')
        plot1.set_ylabel('Temperature')
        plot1_1.set_ylabel('Pressure')
        plt.grid(True)


        ##subplot2

        plt.subplot(323)
        #Age_calcu_age(Temp_age)
        
        for tt in range(Node_age):
            plt.plot(x_grid_age, Argon_record_age_40[tt],'b')
        #print('1 is ',Argon_record_age_40[Node_age-1])
        plt.xlabel('Fraction of $^{39}$Ar')
        plt.ylabel('Concentration of $^{40}$Ar')
        plt.grid(True)



        ##subplot3
        plt.subplot(324)
        plt.plot(x_grid_heat,Argon_record_age_40[Node_age-1],'r')
        for ti in range(Node_heat):
                plt.plot(x_grid_heat, Argon_record_heat_40[ti],'b')
        plt.grid(True)
        plt.xlabel('Fraction of $^{39}$Ar')
        plt.ylabel('Concentration of $^{40}$Ar')

        ##    plt.axis()
        ##    plt.axvline(0.5,color='red',linestyle='--')


        ##subplot4
        plt.subplot(325)
        plt.plot(x_grid_heat,Argon_mode_age_39,'r')
        for ti in range(Node_heat):
                plt.plot(x_grid_heat, Argon_record_heat_39[ti],'b')
        plt.xlabel('Fraction of $^{39}$Ar')
        plt.ylabel('Concentration of $^{39}$Ar')
        plt.grid(True)

        ##subplot6
        plot6=plt.subplot(326)
        plt.plot(total39,age_mode, 'b', linewidth=2,drawstyle='steps-post', label='steps-post')
        for i in range(0,len(total39)-2):
                plot6.add_patch(patches.Rectangle((fuck_fraction[i],(fuck_age[i]-fuck_sigma[i])),(fuck_fraction[i+1]-fuck_fraction[i]),(fuck_sigma[i]*2)))
                #plt.add_patch(fuck_fraction[i],fuck_age[i]+fuck_sigma[i],fuck_fraction[i+1]-fuck_fraction[i],fuck_age[i]-fuck_sigma[i])

        print(total39)
        print(age_mode)
        writer.writerow(fuck_fraction)
        writer.writerow(age_mode)
        
        plt.ylim(0,500)
        plt.xlabel('Fraction of $^{39}$Ar')
        plt.ylabel('Age(Ma)')
        plt.grid(True)
        myfile='E:/'+'data'+str(data_label)+'mode'+str(mode)+'.png'
        plt.savefig(myfile)

csvfile=open('D:\degree-software\data_test.csv', 'w',newline='')
writer = csv.writer(csvfile)





Temp_age,Pressure_age,Temp_heat=dataset3()
data_label=1
runmode(1,Temp_age,Pressure_age,Temp_heat)


Temp_age,Pressure_age,Temp_heat=dataset2()
data_label=2
runmode(1,Temp_age,Pressure_age,Temp_heat)



Temp_age,Pressure_age,Temp_heat=dataset3()
data_label=3
runmode(1,Temp_age,Pressure_age,Temp_heat)



Temp_age,Pressure_age,Temp_heat=dataset4()
data_label=4
runmode(1,Temp_age,Pressure_age,Temp_heat)


Temp_age,Pressure_age,Temp_heat=dataset5()
data_label=5
runmode(1,Temp_age,Pressure_age,Temp_heat)


Temp_age,Pressure_age,Temp_heat=dataset6()
data_label=6
runmode(1,Temp_age,Pressure_age,Temp_heat)


Temp_age,Pressure_age,Temp_heat=dataset7()
data_label=7
runmode(1,Temp_age,Pressure_age,Temp_heat)


Temp_age,Pressure_age,Temp_heat=dataset8()
data_label=8
runmode(1,Temp_age,Pressure_age,Temp_heat)


Temp_age,Pressure_age,Temp_heat=dataset9()
data_label=9
runmode(1,Temp_age,Pressure_age,Temp_heat)


csvfile.close()
##runmode(2,Temp_age,Pressure_age,Temp_heat)
##runmode(3,Temp_age,Pressure_age,Temp_heat)
##
##Temp_age,Pressure_age,Temp_heat=dataset2()
##data_label=2
##runmode(1,Temp_age,Pressure_age,Temp_heat)
##runmode(2,Temp_age,Pressure_age,Temp_heat)
##runmode(3,Temp_age,Pressure_age,Temp_heat)
##
##Temp_age,Pressure_age,Temp_heat=dataset1()
##data_label=1
##runmode(1,Temp_age,Pressure_age,Temp_heat)
##runmode(2,Temp_age,Pressure_age,Temp_heat)
##runmode(3,Temp_age,Pressure_age,Temp_heat)

plt.show()




