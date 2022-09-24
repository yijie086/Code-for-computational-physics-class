n=64000;
h=0.001;
x=zeros(n+1,3);
v=zeros(n+1,3);
x(1,1)=1;
x(1,2)=0;
x(1,3)=0.5;
v(1,1)=-1;
v(1,2)=0.1;
v(1,3)=0.5;
F=zeros(2,3);
for i=1:1:n
   for j=1:1:3
       F(1,j)=-x(i,j)/(x(i,1)^2+x(i,2)^2+x(i,3)^2)^1.5;
       x(i+1,j)=x(i,j)+h*v(i,j)+0.5*F(1,j)*h^2;
   end
   for j=1:1:3
       F(2,j)=-x(i+1,j)/(x(i+1,1)^2+x(i+1,2)^2+x(i+1,3)^2)^1.5;
       v(i+1,j)=v(i,j)+0.5*h*(F(2,j)+F(1,j));
   end
end

x0=zeros(3,1);
x1=zeros(3,1);
for k=1:1:3
    x1(k)=1;
end
plot3(x(:,1),x(:,2),x(:,3),'LineWidth',3);
hold on;
plot3(x0(1),x0(2),x0(3),'o','MarkerSize',10,'MarkerFaceColor','r');
%plot3(x1(1),x1(2),x1(3),'o','MarkerSize',10,'MarkerFaceColor','r');
grid on;
xlabel('x','FontSize',18)
ylabel('y','FontSize',18)
zlabel('z','FontSize',18)
set(gca,'FontSize',18)
legend('分子运动轨迹','力心')
hold off