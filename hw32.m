N=1000;
num=1:1:N;
v=num*0.1;
steps=10.^6;
for i=1:1:steps
    thetai=acos(rand(1,1)*2-1);
    evolutionparticle1=round(rand(1,1)*N+0.5); %random evolution particle 1
    evolutionparticle2=round(rand(1,1)*N+0.5); %random evolution particle 2
    v1=v(1,evolutionparticle1);
    v2=v(1,evolutionparticle2);
    vc=sqrt(v1^2+v2^2+2*v1*v2*cos(thetai))/2;
    thetacout=acos(rand(1,1)*2-1);
    vcr=sqrt(v1^2+v2^2-2*v1*v2*cos(thetai))/2;
    vo1=sqrt(vcr^2+vc^2+2*vcr*vc*cos(thetacout));
    vo2=sqrt(vcr^2+vc^2+2*vcr*vc*cos(pi-thetacout));
    v(1,evolutionparticle1)=vo1;
    v(1,evolutionparticle2)=vo2;
end
hist(v,15);
xlabel('分子速率(m/s)','FontSize',15)
ylabel('分子数量','FontSize',15)
set(gca,'FontSize',15)
hold on;
v=1:1:150;
alpha=1112.7783333333;
f=10.*N.*4.*pi.*v.^2.*(1/(2*pi*alpha)).^(3/2).*exp(-(v.*v)/(2.*alpha));
plot(v,f,'r','LineWidth',2);
legend('程序计算结果','Maxwell分布率');
hold off;