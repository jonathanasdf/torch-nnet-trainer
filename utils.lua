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

