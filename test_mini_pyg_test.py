from pygoruut.pygoruut import Pygoruut

pygoruut = Pygoruut(writeable_bin_dir='')

inputstr = 'Ich baue eine Haus in Sindelfingen'

print('input: ' + inputstr)
result1 = pygoruut.phonemize(language='German', sentence=inputstr)
print('German → IPA (str):', str(result1))

ipa_string = str(result1)

result2 = pygoruut.phonemize(language='German', sentence=ipa_string, is_reverse=True)
print('IPA → English (str):', str(result2))

