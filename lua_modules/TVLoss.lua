--[[
    Copied from https://github.com/DmitryUlyanov/texture_nets

    Copyright Texture Nets Dmitry Ulyanov

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
]]--

require 'nn'

-----Almost copy paste of jcjohnson's code -----------
local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  print('Using TV loss with weight ', strength)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()

  for obj = 1, input:size(1) do
    local input_= input[obj]
    local C, H, W = input_:size(1), input_:size(2), input_:size(3)
    self.x_diff:resize(3, H - 1, W - 1)
    self.y_diff:resize(3, H - 1, W - 1)
    self.x_diff:copy(input_[{{}, {1, -2}, {1, -2}}])
    self.x_diff:add(-1, input_[{{}, {1, -2}, {2, -1}}])
    self.y_diff:copy(input_[{{}, {1, -2}, {1, -2}}])
    self.y_diff:add(-1, input_[{{}, {2, -1}, {1, -2}}])
    self.gradInput[obj][{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
    self.gradInput[obj][{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
    self.gradInput[obj][{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  end

  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)

  return self.gradInput
end