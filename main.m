% This code was developed by Aref Miri Rekavandi for the paper: Rekavandi, A. M., Seghouane, A. K., Boussaid, 
%F., & Bennamoun, M. (2023, June). Extended Expectation Maximization for Under-Fitted Models. In ICASSP 2023-2023 
%IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.
% If you use this code in your study, kindly cite the aforementioned paper.

clc
clear all
close all

%%%%%%%% Initialization %%%%%%%%%%%%
indim=200;              %Dimension
N=4000;                 %Number of samples (Signals)
rank=2;                 %Signal Subspace rank
it=10;                  %Iteration
alpha=0.8;              %Alpha in alpha-divergence
sw=0;                   %Symmetric noise 0:off  1:on
mue1=0;                 %Noise mean for first component
mue2=4;                 %Noise mean for secon component
mue3=10;                %Noise mean for third component
sigma1=1;               %Noise std for first component
sigma2=0.2;             %Noise std for second component
sigma3=0.03;            %Noise std for third component
ep2=0.25;               %Weight of second component 
ep3=0.15;               %Weight of third component 

aaa=alpha;
x=1:indim;
H=zeros(indim,rank);
for i=1:rank
  H(:,i)=cos(2*i*3.14*(1/indim)*x)';
end
H=H-mean(H);
mu1=[zeros(1,N/2) ones(1,N/2)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% With Outliers Scenario %%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Simulations with outliers...')
noise=noise_model(N,indim,ep2,ep3,sigma1,mue1,sigma2,mue2,sigma3,mue3,sw);
figure
[AAA BBB]=hist(noise(:),1000);
plot(BBB,AAA/N,'LineWidth',3)
xlabel('Noise value') 
ylabel('Density')

scale=0.8;
SS=0;
NK=N;
for k=1:N
    TETA1=random('uniform',0.3,0.35,rank,1);
    X(:,k)=H*TETA1;
    Y(:,k)=mu1(k)*scale*X(:,k)+noise(:,k);
end

space=H;
for ind=1:N
     inte=rand(rank,1);
     thetahat=inte;
     temp=Y(:,ind);
     temp=5*(temp-mean(temp))/sqrt(var(temp-mean(temp)));
  
     K=2;
     mu=[-1;1];
     sigma2=[1;1];
     w=rand(1,K);
     w=w/sum(w);
     
     
     mukl=mu;
     sigma2kl=sigma2;
     wkl=w;

 for i=1:it
    ins=temp-space*thetahat;
    for j=1:indim
        for r=1:K
            d=likelihood(mukl,sigma2kl,wkl,ins(j),1);
            membership(j,r)=(wkl(r)/sqrt(2*pi*sigma2kl(r)))*exp(-0.5*(sigma2kl(r)^(-1))*(ins(j)-mukl(r))^2)/d;
        end
    end
    JJ=zeros(indim,indim,K);
    for j=1:K    
        mukl(j)=((sum(membership(:,j)))^(-1))*membership(:,j)'*ins;
        sigma2kl(j)=((sum(membership(:,j)))^(-1))*membership(:,j)'*(ins-mukl(j)).^2;
        wkl(j)=sum(membership(:,j))/indim;
        for t=1:indim
            JJ(t,t,j)=membership(t,j)/sigma2kl(j);
        end
    end
    J=zeros(indim,indim);
    for j=1:K
        J=J+JJ(:,:,j);
    end
    thetahat=zeros(rank,1);
    for j=1:K
        thetahat=thetahat+((H'*J*H)^(-1))*H'*(JJ(:,:,j)*(ins-mukl(j)));
    end

 end       
          
   EM(ind)=thetahat'*thetahat;
  
   
   
     thetahat=inte;
     mualpha=mu;
     sigma2alpha=sigma2;
     walpha=w;
     
 for i=1:it
    ins=temp-space*thetahat;
    for j=1:indim
        for r=1:K
            d=likelihood(mualpha,sigma2alpha,walpha,ins(j),alpha);
            membership(j,r)=(walpha(r)/sqrt(2*pi*sigma2alpha(r)))*exp(-0.5*(sigma2alpha(r)^(-1))*(ins(j)-mualpha(r))^2)/d;
        end
    end
    JJ=zeros(indim,indim,K);
    for j=1:K    
        mualpha(j)=((sum(membership(:,j)))^(-1))*membership(:,j)'*ins;
        sigma2alpha(j)=((sum(membership(:,j)))^(-1))*membership(:,j)'*(ins-mualpha(j)).^2;
        walpha(j)=sum(membership(:,j))/indim;
        for t=1:indim
            JJ(t,t,j)=membership(t,j)/sigma2alpha(j);
        end
    end
    J=zeros(indim,indim);
    for j=1:K
        J=J+JJ(:,:,j);
    end
    thetahat=zeros(rank,1);
    for j=1:K
        thetahat=thetahat+((H'*J*H)^(-1))*H'*(JJ(:,:,j)*(ins-mualpha(j)));
    end

 end       
          
   EMi(ind)=thetahat'*thetahat;
     
   
  thetahat=((H'*H)^(-1))*H'*temp;
  LS(ind)=thetahat'*thetahat;
     
         
end
fprintf('Done!')
fprintf('\n')
i=1;
last=(max([EM,EMi,LS]));
starting=0;
step=(last-starting)/10000;
for th=starting:step:last
    pd1(i)=(200*(sum(EM(N/2+1:N)>th)))/N;
    pf1(i)=(200*(sum(EM(1:N/2)>th)))/N;
    
    pd2(i)=(200*(sum(EMi(N/2+1:N)>th)))/N;
    pf2(i)=(200*(sum(EMi(1:N/2)>th)))/N;
    
    pd3(i)=(200*(sum(LS(N/2+1:N)>th)))/N;
    pf3(i)=(200*(sum(LS(1:N/2)>th)))/N;
    
    i=i+1;
end
figure
subplot(2,2,1)
 plot([EM(1:N/2) zeros(1,N/2)])
 title('EM')
 hold on
 plot([zeros(1,N/2) EM(N/2+1:N)])
 subplot(2,2,2)
 plot([EMi(1:N/2) zeros(1,N/2)])
 title('EMi')
 hold on
 plot([zeros(1,N/2) EMi(N/2+1:N)])
 subplot(2,2,3)
 plot([LS(1:N/2) zeros(1,N/2)])
 title('LS')
 hold on
 plot([zeros(1,N/2) LS(N/2+1:N)])

%%%%%%%%%%%%%%%%%%%%% without outliers Case %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Simulations without outliers...')
sw=0;
mue1=0;
mue2=4; 
mue3=10;
sigma1=1;
sigma2=0.2;
sigma3=0.03;
ep2=0.25;
ep3=0;
noise=noise_model(N,indim,ep2,ep3,sigma1,mue1,sigma2,mue2,sigma3,mue3,sw);


scale=0.8;
SS=0;
NK=N;
for k=1:N
    TETA1=random('uniform',0.3,0.35,rank,1);
    X(:,k)=H*TETA1;
    Y(:,k)=mu1(k)*scale*X(:,k)+noise(:,k);
end

space=H;
for ind=1:N
     inte=rand(rank,1);
     thetahat=inte;
     temp=Y(:,ind);
     temp=5*(temp-mean(temp))/sqrt(var(temp-mean(temp)));
  
     K=2;
     mu=[-1;1];
     sigma2=[1;1];
     w=rand(1,K);
     w=w/sum(w);
     
     
     mukl=mu;
     sigma2kl=sigma2;
     wkl=w;

 for i=1:it
    ins=temp-space*thetahat;
    for j=1:indim
        for r=1:K
            d=likelihood(mukl,sigma2kl,wkl,ins(j),1);
            membership(j,r)=(wkl(r)/sqrt(2*pi*sigma2kl(r)))*exp(-0.5*(sigma2kl(r)^(-1))*(ins(j)-mukl(r))^2)/d;
        end
    end
    JJ=zeros(indim,indim,K);
    for j=1:K    
        mukl(j)=((sum(membership(:,j)))^(-1))*membership(:,j)'*ins;
        sigma2kl(j)=((sum(membership(:,j)))^(-1))*membership(:,j)'*(ins-mukl(j)).^2;
        wkl(j)=sum(membership(:,j))/indim;
        for t=1:indim
            JJ(t,t,j)=membership(t,j)/sigma2kl(j);
        end
    end
    J=zeros(indim,indim);
    for j=1:K
        J=J+JJ(:,:,j);
    end
    thetahat=zeros(rank,1);
    for j=1:K
        thetahat=thetahat+((H'*J*H)^(-1))*H'*(JJ(:,:,j)*(ins-mukl(j)));
    end

 end       
          
   EM1(ind)=thetahat'*thetahat;
  
   
   
     thetahat=inte;
     mualpha=mu;
     sigma2alpha=sigma2;
     walpha=w;
     
 for i=1:it
    ins=temp-space*thetahat;
    for j=1:indim
        for r=1:K
            d=likelihood(mualpha,sigma2alpha,walpha,ins(j),alpha);
            membership(j,r)=(walpha(r)/sqrt(2*pi*sigma2alpha(r)))*exp(-0.5*(sigma2alpha(r)^(-1))*(ins(j)-mualpha(r))^2)/d;
        end
    end
    JJ=zeros(indim,indim,K);
    for j=1:K    
        mualpha(j)=((sum(membership(:,j)))^(-1))*membership(:,j)'*ins;
        sigma2alpha(j)=((sum(membership(:,j)))^(-1))*membership(:,j)'*(ins-mualpha(j)).^2;
        walpha(j)=sum(membership(:,j))/indim;
        for t=1:indim
            JJ(t,t,j)=membership(t,j)/sigma2alpha(j);
        end
    end
    J=zeros(indim,indim);
    for j=1:K
        J=J+JJ(:,:,j);
    end
    thetahat=zeros(rank,1);
    for j=1:K
        thetahat=thetahat+((H'*J*H)^(-1))*H'*(JJ(:,:,j)*(ins-mualpha(j)));
    end

 end       
          
   EMi1(ind)=thetahat'*thetahat;
     
   
  thetahat=((H'*H)^(-1))*H'*temp;
  LS1(ind)=thetahat'*thetahat;
     

end
 fprintf('Done!')
 fprintf('\n')
 fprintf('Now wait for the plots!')

i=1;
last=(max([EM1,EMi1,LS1]));
starting=0;
step=(last-starting)/10000;
for th=starting:step:last
    pd11(i)=(200*(sum(EM1(N/2+1:N)>th)))/N;
    pf11(i)=(200*(sum(EM1(1:N/2)>th)))/N;
    
    pd22(i)=(200*(sum(EMi1(N/2+1:N)>th)))/N;
    pf22(i)=(200*(sum(EMi1(1:N/2)>th)))/N;
    
    pd33(i)=(200*(sum(LS1(N/2+1:N)>th)))/N;
    pf33(i)=(200*(sum(LS1(1:N/2)>th)))/N;
    
    i=i+1;
end



figure
plot(pf11,pd11,'b','LineWidth',2)
hold on
plot(pf22,pd22,'r','LineWidth',2)
hold on
plot(pf33,pd33,'g','LineWidth',2)
hold on 
plot(pf1,pd1,'b--','LineWidth',2)
hold on
plot(pf2,pd2,'--r','LineWidth',2)
hold on
plot(pf3,pd3,'--g','LineWidth',2)
hold on

grid on

    legend({'EM','EMi with $\alpha=0.8$','Least Square','EM in outliers','EMi in outliers and $\alpha=0.8$','Least Square in outliers'}, ...
        'Interpreter', 'LaTeX')
    xlabel('Probability of False Alarm (\%)', 'Interpreter', 'LaTeX')
    ylabel('Probability of Detection (\%)', 'Interpreter', 'LaTeX')
    title('', 'FontName', 'Times New Roman', ...
        'FontSize',10,'Color','k', 'Interpreter', 'LaTeX')
% print('Performance-P0080','-depsc')    
     