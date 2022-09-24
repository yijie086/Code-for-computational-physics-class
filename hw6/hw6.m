N=90;
omega=7/4;
x=0:pi/N:pi;
y=0:pi/N:pi;
phi=zeros(N+1);
phi(:,N+1)=sin(x);
temp=0;
for n=1:1:1000
    for i=2:1:N
        for j=2:1:N
            temp=(phi(i+1,j)+phi(i,j+1)+phi(i-1,j)+phi(i,j-1))/4;
            phi(i,j)=(1-omega)*phi(i,j)+omega*temp;
        end
    end
end
figure(1)
surf(x,y,phi')
xlabel('x','FontSize',18)
ylabel('y','FontSize',18)
zlabel('\phi','FontSize',18)
set(gca,'FontSize',18)
figure(2)
phithe=sin(x)'*sinh(y)/sinh(pi);
surf(x,y,phithe')
xlabel('x','FontSize',18)
ylabel('y','FontSize',18)
zlabel('\phi','FontSize',18)
set(gca,'FontSize',18)
figure(3)
delta=abs(phi(:,:)-phithe(:,:));
surf(x,y,delta')
xlabel('x','FontSize',18)
ylabel('y','FontSize',18)
zlabel('\delta','FontSize',18)
set(gca,'FontSize',18)
