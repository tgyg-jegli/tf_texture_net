require 'nn'
require 'lua_modules/InstanceNormalization'
require 'lua_modules/TVLoss'

local cmd = torch.CmdLine()

cmd:option('--model', '', 'Path to trained model.')
local params = cmd:parse(arg)

-- Load model
local model = torch.load(params.model)

-- Print
print('List of layers')
print('#######################################################')
print(model)

print('')

print('List details of layers')
print('#######################################################')
for key,value in pairs(model) do print(key,value) end

print('')

print('List details of first res_block (model.modules[12])')
print('#######################################################')
for key,value in pairs(model.modules[12]) do print(key,value) end


print('')

print('List details of first res_block (model.modules[12].modules[1].modules[2])')
print('#######################################################')
for key,value in pairs(model.modules[12].modules[1].modules[2]) do print(key,value) end