"""
Acak urutan object di testfile.xml sambil mempertahankan urutan posisi (z-sequence).
Hasilnya: kelas plastic/paper/metal/glass/elektronik tercampur acak.
"""
import re
import random

XML_PATH = r'Mojoco\testfile.xml'

with open(XML_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Temukan semua body element objek (dari <body name="..." hingga </body>)
pattern = re.compile(
    r'(<body name="(?:plastic|paper|metal|glass|elektronik)\d+[^"]*".*?</body>)',
    re.DOTALL
)

matches = list(pattern.finditer(content))
bodies  = [m.group(1) for m in matches]

print(f'Ditemukan {len(bodies)} object body')

# Ambil posisi dari setiap body (hanya atribut pos pertama = posisi body itu sendiri)
pos_re   = re.compile(r'pos="([^"]+)"')
positions = []
for b in bodies:
    m = pos_re.search(b)
    positions.append(m.group(1) if m else '')

# Shuffle bodies (posisi tetap urut, konten diacak)
random.seed()          # acak setiap kali
shuffled = bodies[:]
random.shuffle(shuffled)

def replace_first_pos(body_text, new_pos):
    return pos_re.sub(f'pos="{new_pos}"', body_text, count=1)

# Pasangkan body acak ke posisi urutan asli
final_bodies = [replace_first_pos(b, p) for b, p in zip(shuffled, positions)]

# Bangun konten baru: ganti span asli dengan body acak
result = []
prev = 0
for m, new_body in zip(matches, final_bodies):
    result.append(content[prev:m.start()])
    result.append(new_body)
    prev = m.end()
result.append(content[prev:])

new_content = ''.join(result)

with open(XML_PATH, 'w', encoding='utf-8') as f:
    f.write(new_content)

# Verifikasi: tampilkan 5 pertama
check = pattern.findall(new_content)
print('5 object pertama setelah shuffle:')
for b in check[:5]:
    name = re.search(r'name="([^"]+)"', b).group(1)
    pos  = pos_re.search(b).group(1)
    print(f'  {name:20s} pos={pos}')

print('Selesai.')
