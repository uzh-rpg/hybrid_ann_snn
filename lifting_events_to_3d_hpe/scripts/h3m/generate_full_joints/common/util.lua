
local M = {}

function M.dir(path, ext)
  list = {}
  for file in paths.files(path) do
    if file:find(ext .. '$') then
      table.insert(list, file)
    end
  end
  table.sort(list)
  return list
end

function M.makedir(dir)
  if not paths.dirp(dir) then
    paths.mkdir(dir)
  end
end

function M.keys(table)
  local keys={}
  local i = 0
  for k in pairs(table) do
    i = i+1
    keys[i] = k
  end
  return keys
end

function M.unique(input)
-- input should be a one dimensinoal tensor or a table
  local N
  if torch.type(input) ~= 'table' then
    N = input:numel()
  else
    N = #input
  end
  local b = {}
  for i = 1, N do
    b[input[i]] = true
  end
  local out = {}
  for i in pairs(b) do
    table.insert(out, i)
  end
  table.sort(out)
  if torch.type(input) ~= 'table' then
    out = torch.Tensor(out)
  end
  return out
end

function M.find(X, n)
-- X should be a one dimensional tensor
  local indices = torch.linspace(1,X:size(1),X:size(1)):long()
  return indices[X:eq(n)]
end

function M.setdiff(A, B)
-- input should be two one dimensinoal tensor
  local out = {}
  for i = 1, A:numel() do
    if M.find(B, A[i]):numel() == 0 then
      table.insert(out, A[i])
    end
  end
  return torch.Tensor(out)
end

function M.append(tab1, tab2)
  local t = {}
  for i = 1, #tab1 do
    t[i] = tab1[i]
  end
  for i = 1, #tab2 do
    t[#tab1+i] = tab2[i]
  end
  return t
end

function M.slice(tab, ind)
  local t1, t2 = {}, {}
  for i = 1, ind do
    table.insert(t1, tab[i])
  end
  for i = ind+1, #tab do
    table.insert(t2, tab[i])
  end
  return t1, t2
end

-- Calls provides function on each entry of table t
function M.applyTab(fn, t)
  assert(type(t) == 'table')
  local t_ = {}
  for i = 1, #t do
    t_[i] = fn(t[i])
  end
  return t_
end

function M.strsplit(str, delimiter)
  if delimiter == nil then
    delimiter = "%s"
  end
  local split = {}
  for str in string.gmatch(str,"([^" .. delimiter .."]+)") do
    table.insert(split, str)
  end
  return split
end

return M