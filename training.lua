-- inspired from github.com/soumith/dcgan.torch
	
require 'torch'
require 'nn'
require 'optim'
-- various otions for training
opt = {
	gpu = 1, -- by default uses it
	num_epoch = 1000,  -- ??
	batch_size = 1, -- ??
	ntrain = 100, -- ??
  display_id = 10, -- ??????
  name = "Experiment",
  nThreads = 4,
  dataset = 'lsun',
  niter = 25, --- ???
  momentum = .5, -- ????
  lr = .0001, -- ???
  noise = 'normal',
  fineSize = 64,
  batchSize = 1,
  nz = 100, -- ?????
  ngf = 64, --  ??
  display = 1,
  ndf = 64,  -- ???
}
for field,value in ipairs(opt) do
	opt[field] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
end

if opt.display == 0 then opt.display = false
end
if opt.display then disp = require 'display' end
opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())


--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------


--	Initializing weights for the network
local function init_weights(model)
	local name_model = torch.type(model)
	if name:find('Convolution') then
		m.weight:normal() -- define them later 
		m:noBias() -- try using biases later
	elseif name:find('BatchNormalization') then
	  if m.weight then m.weight:normal() end -- define later
	  if m.bias then m.bias:fill(0) end -- define later
	end
end

local real_label = 1
local fake_label = 0
local noise = torch.Tensor(opt.batchSize,nz,1,1)

local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

local errD, errG
local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local epoch_tm = torch.Timer()
local batch_tm = torch.Timer()
local data_tm = torch.Timer()

optimStateD = {
	learningRate = opt.lr,
  beta1 = opt.beta1,
}
optimStateG = {
  learningRate = opt.lr,
  beta1 = opt.beta1,	
}

if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();
   netG:cuda();
   criterion:cuda()
end


local parametersD,gradParametersD = netD:getParameters()
local parametersG,gradParametersG = netG:getParameters() 

-----------------------------------------
-- Models -------------------------------
-----------------------------------------

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

--Generator------------------------------
local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)


--Discriminator----------------------------
local netD = nn.Sequential()
-- ????????????????????
-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)


---Criterion-----------------------------
local criterion = nn.BCECriterion()
local criterion_absolute = nn.AbsCriterion()

--- Gradient_Calculators-----------------
local fDx = function(x)
  gradParametersD:zero()
  data_tm:reset()
  local real = data:getBatch()
  data_tm:stop()
  input:copy(real)
  label:fill(real_label)

  local output = netD:forward(input)
  local errD_real = criterion:forward(input,label)
  local df_by_do = criterion:backward(output,label) -- find the d(function)/d(output)
  netD:backward(input, df_by_do) -- do back propogation
  if opt.noise == 'uniform' then 
  	noise:uniform(-1,1)
  else if opt.noise == 'normal then' then
  	noise:normal(0,1)
  end
  local fake = netG:forward(noise)
  input:copy(fake)
  label:fill(fake_label)
  local output = netD:forward(input)
  local errD_fake = criterion:forward(input,label)
  local df_by_do = criterion:backward( output, label)
  netD:backward(input,df_by_do)
  errD = errD_real + errD_real
  return err_D,gradParametersD
end

local fGx = function(x)
  gradParametersG:zero()
  label:fill(real_label)
  local output = netD.output
  errG = criterion:forward(output,real_label)  
  local df_by_do = criterion:backward()
  local df_by_dg = netD:updateGradInput(input,df_by_do)

  netG:backward(noise,df_by_dg)
  return errG,gradParametersD
end

----------------------------------
-- training	
-----------------------------------------
noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

for epoch =1,opt.num_epoch do 	epoch_tm:reset()
	local count = 0
	for i = 1,math.min(data:size(),opt.ntrain),opt.batch_size do 
		batch_tm:reset()
		-- Update the Discriminator network and the Generator networks
		optim.adam(fDx, parametersD, optimStateD)
    optim.adam(fGx, parametersG, optimStateG)
    counter = counter + 1
    if counter % 10 == 0 and opt.display_images == 1 then 
      local fake = netG:forward(noise_input)
      local real = data:getBatch()
      disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
    end
    if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
end