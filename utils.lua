function defineBaseOptions(cmd)
  cmd:option('-nThreads', 8, 'number of threads')
  cmd:option('-gpu', "", 'comma separated list of GPUs to use')
  cmd:option('-nGPU', 4, 'number of GPU to use. Ignored if gpu is set')
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function string:split(sep)
  local sep, fields = sep or ",", {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(c) fields[#fields+1] = c end)
  return fields
end

function findModuleByName(model, name)
  for i=1,#model.modules do
    if model.modules[i].name == name then
      return model.modules[i]
    end
  end
  return nil
end

local va = require "vararg"
function bind(f, ...)
  local outer_args = va(...)
  local function closure (...)
    return f(va.concat(outer_args, va(...)))
  end
  return closure
end
