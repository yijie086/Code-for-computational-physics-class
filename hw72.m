N=64;
step=2000;
h=0.02;
x=zeros(N,3);
x1=zeros(N,3);
xr=zeros(N,3);
flag=0;
E=zeros(step,1);
T=2;
rc=5;

l=1;
while(l<N+1)
    flag=0;
    x(l,:)=round(0.5+9*rand(1,3));
    for k=1:1:l-1
       if (x(k,1)==x(l,1))&&(x(k,2)==x(l,2))&&(x(k,3)==x(l,3))
          flag=1;
       end
    end
    if flag==1
        l=l-1;
    end
    l=l+1;
end
v=(2*rand(N,3)-1);
F1=zeros(64,3);
F2=zeros(64,3);

for i=1:1:step
   for n=1:1:N
       for r=1:1:3
           F1(n,r)=0;
           F2(n,r)=0;
       end
       for k=1:1:N
           if k~=n
               for m=1:1:3
                    if x(n,m)-x(k,m)>5
                        xr(k,m)=x(k,m)+10;
                        xr(k,m)=xr(k,m)-x(n,m);
                    elseif x(n,m)-x(k,m)<=-5
                        xr(k,m)=x(k,m)-10;
                        xr(k,m)=xr(k,m)-x(n,m);
                    else
                        xr(k,m)=x(k,m)-x(n,m);
                    end
               end
               for m=1:1:3
                   F1(n,m)=F1(n,m)-48*xr(k,m)/(xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)^7+24*xr(k,m)/(xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)^4;
                   if (xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)>rc^2
                       F1(n,m)=F1(n,m)+48*xr(k,m)/(xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)^7-24*xr(k,m)/(xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)^4;
                   end
               end
           end
       end
       for m=1:1:3
          x1(n,m)=mod(x(n,m)+h*v(n,m)+0.5*F1(n,m)*h^2,10); 
       end
   end
   
   for n=1:1:N
       for k=1:1:N
           if k~=n
               for m=1:1:3
                    if x1(n,m)-x1(k,m)>5
                        xr(k,m)=x1(k,m)+10;
                        xr(k,m)=xr(k,m)-x1(n,m);
                    elseif x1(n,m)-x1(k,m)<=-5
                        xr(k,m)=x1(k,m)-10;
                        xr(k,m)=xr(k,m)-x1(n,m);
                    else
                        xr(k,m)=x1(k,m)-x1(n,m);
                    end
               end
               for m=1:1:3
                    F2(n,m)=F2(n,m)-48*xr(k,m)/(xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)^7+24*xr(k,m)/(xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)^4;
                    if (xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)>rc^2
                        F2(n,m)=F2(n,m)+48*xr(k,m)/(xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)^7-24*xr(k,m)/(xr(k,1)^2+xr(k,2)^2+xr(k,3)^2)^4;
                    end
               end
           end
       end
       for m=1:1:3
          v(n,m)=v(n,m)+0.5*(F1(n,m)+F2(n,m))*h; 
       end
   end
   beta=0;
   for n=1:1:N
       for m=1:1:3
          beta=beta+v(n,m)^2; 
       end
   end
   beta=(T*(N-1)/(beta*16))^0.5;
   for n=1:1:N
       for m=1:1:3
          v(n,m)=beta*v(n,m); 
       end
   end
   for n=1:1:N
      for m=1:1:3
         x(n,m)=x1(n,m); 
      end
   end
   for n=1:1:N
       for m=1:1:3
          E(i)=E(i)+v(n,m)^2; 
       end
   end
   E(i)=E(i)/2;
     %  figure(1)
     %  plot3(v(:,1),v(:,2),v(:,3),'o','MarkerSize',5,'MarkerFaceColor','r')
     %  xlim([-0.4,0.4])
     %  ylim([-0.4,0.4])
     %  zlim([-0.4,0.4])
     %  grid on
     %  figure(2)
     %  plot3(x(:,1),x(:,2),x(:,3),'o','MarkerSize',5,'MarkerFaceColor','r')
     %  xlim([0,10])
     %  ylim([0,10])
     %  zlim([0,10])
     %  grid on
     %  pause(0.01)
       
end
figure(1)
plot3(v(:,1),v(:,2),v(:,3),'o','MarkerSize',5,'MarkerFaceColor','r')
xlim([-1,1])
ylim([-1,1])
zlim([-1,1])
xlabel('V_x')
ylabel('V_y')
zlabel('V_z')
set(gca,'FontSize',14)
grid on

figure(2)
plot3(x(:,1),x(:,2),x(:,3),'o','MarkerSize',5,'MarkerFaceColor','r')
xlim([0,10])
ylim([0,10])
zlim([0,10])
xlabel('x')
ylabel('y')
zlabel('z')
set(gca,'FontSize',14)
grid on