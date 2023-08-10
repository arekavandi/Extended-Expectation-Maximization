function out=likelihood(mu,sigma2,w,noise,alpha)

ts=0;
k=max(size(mu));
for i=1:k
ts=ts+(w(i)/sqrt(2*pi*sigma2(i)))*exp(-0.5*(sigma2(i)^(-1))*(noise-mu(i))^2);
end

out=(ts)^alpha;


end