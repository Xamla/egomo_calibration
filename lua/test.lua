local xamla3d =require('./xamla3d')

local test = {}
table.insert(test, torch.DoubleTensor(3,1):zero())
table.insert(test, torch.DoubleTensor({1,0,0}):view(3,1))
table.insert(test, torch.DoubleTensor({0,1,0}):view(3,1))
table.insert(test, torch.DoubleTensor({0,0,1}):view(3,1))


for i = 1,10000 do 
  table.insert(test, torch.rand(3,1))
end


for i =1, #test do
  v = test[i]
  R = xamla3d.axisAngleToRotMatrix(v)
  vd = xamla3d.rotMatrixToAxisAngle(R)
  print(torch.norm(v - vd))
end
