function out=noise_model(N,dim,ep1,ep2,sigma1,mu1,sigma2,mu2,sigma3,mu3,sw)
n=N*dim;

pn=floor((1-ep1-ep2)*n);
sn1=floor((ep1)*n);
sn2=n-pn-sn1;

noise1=random('normal',mu1,sigma1,pn,1);
noise2=random('normal',mu2,sigma2,sn1,1);
noise3=random('normal',mu3,sigma3,sn2,1);
if sw==1;
    noise2=abs(noise2);
    noise3=abs(noise3);
end

noise=[noise1;noise2;;noise3];
out=shuffle(noise);


out=reshape(out,dim,N);


 function v=shuffle(v)
     v=v(randperm(length(v)));
 end
end