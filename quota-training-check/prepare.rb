root = "/home/nate/Desktop/recipe-dev"
destination = "/home/nate/claude/quota-training-check.yJjrH1"
File.open("#{root}/corpus-clean.tsv") do |input|
  File.open("#{destination}/corpus-clean.tsv", "w") do |output|
    output.write(input.gets)
    8.times { output.write(input.gets || abort("Insufficient rows")) }
  end
end
script = File.read("#{root}/quota_rat.rs")
script = script.sub('.epochs(100000)', '.epochs(10)')
script = script.sub('.rat("./evaluate-quota")', ".rat(\"#{root}/evaluate-quota\")")
File.write("#{destination}/quota_rat.rs", script)
