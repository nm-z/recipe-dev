root = File.dirname(__FILE__)
old = File.open('/home/nate/codex/quota-policy-benchmark/full-vector.iDFL4FYi/evaluations.tsv')
count = 0
overlap = 0
maximum = 0.0
best = nil
File.foreach(File.join(root, 'evaluations.tsv')) do |line|
  fields = line.chomp.split("\t")
  count += 1
  raise "index mismatch at #{count}" unless fields[0] == "call_index=#{count}"
  values = fields[1..3].map { |field| Float(field.split('=', 2)[1]) }
  raise "nonfinite metrics at #{count}" unless values.all?(&:finite?)
  if best.nil? || values[1] > best[1]
    best = [count, values[1], values[0], fields[4]]
  end
  if previous = old.gets
    prior = previous.chomp.split("\t")
    raise "proposal mismatch at #{count}" unless prior[4] == fields[4]
    values.zip(prior[1..3]).each do |value, text|
      prior_value = Float(text.split('=', 2)[1])
      difference = (value - prior_value).abs
      maximum = [maximum, difference].max
      raise "metric mismatch at #{count}" if difference > 1e-9 * [1.0, prior_value.abs].max
    end
    overlap += 1
  end
end
old.close
raise "wrong count #{count}" unless count == 770_493
workers = File.readlines(File.join(root, 'workers.tsv')).drop(1).map { |line| line.chomp.split("\t") }
raise 'missing device work' unless workers.length == 11 && workers.all? { |row| row[2] == 'true' && row[4].to_i > 0 }
raise 'worker count differs' unless workers.sum { |row| row[4].to_i } == count
raise 'run failed' unless File.read(File.join(root, 'exit-status.txt')).strip == '0'
raise 'model missing' unless File.size?(File.join(root, 'quota-model.ogdl'))
puts "rows=#{count} workers=#{workers.length} previous_scores_checked=#{overlap} max_metric_difference=#{maximum}"
puts "best_quota_r2=#{best[1]} at_record=#{best[0]} raw_score=#{best[2]}"
