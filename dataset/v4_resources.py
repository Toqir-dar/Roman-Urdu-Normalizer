"""Hand-written resources for the v4 Roman Urdu normalization dataset (build_v4.py).

v4 conventions follow the spellings of the large templated part of external_raw/dataset.csv (nahi, mein, main,
hoon, yeh, woh, to, ke, kya / kiya, bohat, acha, hai, hain), because that is where most of the clean gold comes from.
Earlier versions (v2 / v3) used nahin, toh, keh; those are remapped with CONVENTION below.
"""

# ------------------------------------------------------------------ canonical spelling conventions
# old canonical -> v4 canonical, applied to every gold token from every source (Dakshina, UCI draft, lexicon)
CONVENTION = {
    "nahin": "nahi", "toh": "to", "keh": "ke", "ami": "ammi", "abu": "abbu", "hun": "hoon", "wo": "woh",
    "ye": "yeh", "thoda": "thoda", "bazaar": "bazar", "saath": "sath",
}
# junk v3 lexicon canonicals (alignment artefacts) removed before the lexicon is used
DROP_V3_CANON = set("main nahi hun wo ye abu".split())   # main=مین, nahi=نہی, ye=ے ... (v4 re-adds main/nahi properly)

# ------------------------------------------------------------------ templated-data entities (masked to build frames)
PEOPLE = "ali ahmed ayesha hamza sana mahnoor usman fatima bilal hina zara danish imran sara".split()
CITIES = """quetta sialkot peshawar rawalpindi gujranwala bahawalpur karachi faisalabad islamabad lahore multan
hyderabad""".split()
APPS = "whatsapp instagram twitter telegram linkedin tiktok facebook youtube snapchat careem uber".split()
SUBJECTS = "physics statistics math history chemistry economics biology english urdu computer".split()
NUMBERS = "ek do teen chaar paanch chhe saat aath nau das sau hazaar".split()
TIME_WORDS = "aaj kal abhi subah shaam raat dopahar parson".split()
TIME_HEADS = "pichle agle is har".split()           # + saal / hafte / mahine / shaam ...
TIME_TAILS = "saal hafte mahine".split()
KIN = "ammi abbu behen bhai chacha mamu nana nani dada dadi phuppo teacher shopkeeper".split()
FOODS = """biryani pulao nihari haleem samosa paratha kebab shawarma burger pizza sandwich omelette fries juice chai
coffee daal chawal roti sabzi karahi raita cake masala namak masale""".split()

# English words that appear in the templated data (lang = EN; kept as written)
ENGLISH_SYN = """order restaurant the laptop mobile ticket call complete phone mall online update bus university class
gift presentation printer late keyboard bike headphone office delete charger camera gym submit homework bag result
video login famous message notification hotel password comment messages notifications app problems post recipe reels
price plaza portfolio free proposal thesis worksheet backup quality miss deadline rate please bags gifts announce spicy
router feel screenshot toys option bottle sir excited traffic parcel follow delivery error download fruits change
classes question syllabus link coaster chapter active friend requests errors uninstall problem export block chapters
borrow program photo antivirus revision posts emails present battery exam game sale movie useful pen account cafe
important email nervous flight semester internet discount delay decide hurt discuss meeting slides status payment card
deliver reply tension lock slow percent formulas rupees otp file fruit location mistakes wifi busy voice signal funny
academy access ad airport area assignment back bill book books bore brand browser cancel charge chat check code college
computer control copy course crash cup data database date department discussion document driver exchange extra family
fees files folder form furniture glass group hack hall hang hospital idea install last lecture library light list load
map market marks minute monitor mouse note notes number open package paper park pay petrol picture plan practical
practice profile project quiz ready report reset return rickshaw road school score section server settings shoes
software speed start station stock story students survey system tablet tag taste taxi teacher test time tool train
training try upload use van version viral website""".split()

# common English words for social text (with ENGLISH_SYN): used to tag EN and to detect English context
ENGLISH_COMMON = """a about above after again against all also always am amazing an and another any anyone anything
are aren't around as ask at away awesome baby bad be beautiful because been before being best better between big
both boy brother but by can can't cannot care channel character come comedy cool could country cricket cute dear did
didn't do does doesn't doing don't done down drama during each end enjoy episode even ever every everyone everything
fake fan fans favorite favourite few final first for friends from full funny get girl girls give go god going gone
good got great guys had has hate have he hello help her here hero hey him his hit hope how i i'm if in india indian
into is isn't it it's its just keep kind know last least let life like line little live long look looking lot love
lovely lyrics made make man many match matter may me mean men might more most much music must my myself need never
new news next nice no not nothing now of off oh ok okay old on once one only or other our out over people perfect
person play player please pls plz pm pretty proud real really right role sad said same say see seen she should show
sing singer sir so some someone something song songs sorry still stop such super support sure team than thank
thanks that that's the their them then there these they thing things think this those though through to today
together too true try two under up us very voice want was watch way we well were what when where which while who
why will with without woman women won't work world would wow wrong yeah yes yet you you're your yours lol omg btw
dont cant im ive thats whats u ur r nd bro guys sis plzz pls thx tnx ok okk okay bless respect wish wishes congrats
congratulations birthday happy morning night amazing awesome brilliant excellent fantastic legend legendary genius
masterpiece outstanding heart hearts heartbreaking heart-touching touching emotional actor actress acting director
movie movies film films series season serial dramas episode episodes video videos views subscribe subscribed
subscribers like likes comment comments share shared post posts page link status story stories official channel
army police court judge case government govt minister prime president party leader election elections vote votes
nation national public media news report reporter anchor interview show host guest audience
""".split()

# ------------------------------------------------------------------ social text: curated noisy -> canonical map
# Written from the 1,500 most frequent word types of the real comments in dataset.csv. Context-dependent words
# (main/mein, he, han, or, bad, chor, ...) are handled by rules in build_v4.py, not here.
# Format: "canonical: variant variant ..."
SOCIAL_MAP_RAW = """
ke: ky kay kae k kei
ki: kee
ka: kaa
ko: ku_ko
yeh: yai ye yh yeh yee yea_ur
yahi: yehi yhi yahee
wahi: wohi wahee
woh: wo vo voh wuh
ne: ny nay ne
se: sy sey se
koi: koe koii koyi koyee
nahi: ni nhi nahin nai nh nhe nehi nahee nahin nhin nahiii nae nahe
bhi: b bi bhee bhe bh v
bohat: bohot bahut bhot bhut buhat buht bahot bht boht bohut bohooot bhout bhohot bhot bahoot bhoht
hai: hy hay hae h hei hae hy hai
hain: hn hen hein hyn hain hein haen hayn
hoon: hun hu hon hoo
de: dy
dein: dain daen dein den_ur
diya: deya dia diya diyaa
diye: diey deay diye
dena: daina dena deena
dete: dety detay
liya: lia leya liya
liye: lye leye liay liy lie_ur liey liyay
le: lay le
lena: laina lena
lene: layne laine
lein: lain lein
gaya: gya gia gaya
gayi: gai gyi gae gayi
gaye: gye gaey gaye
ge: gy
hua: huwa hova hua
hui: hoi hui
hue: huay huye howy hoye hue
hone: honay hony
the: thay thy thay
tha: th tha
to: tou tw tho tau
kaise: kaisay kesy kesay kese kaisy kesay kisay_x
aisa: esa asa aisa
aise: aisay esy asy aisay aese
aisi: asi esi aisi
waise: waisay wesay wese wesy
waisi: wesi
jaise: jese jesy jaisay
jaisi: jesi jaisi
kaun: kon kn kaun
kaunsa: konsa
kaunsi: konsi
aap: ap
aapko: apko
aapki: apki
aapka: apka
aapke: apke apky
apne: apnay apny
rahe: rahay rhy rahey rhe rahy rahe
raha: rha rha
rahi: rhi rahee
karte: kartay krty krte karty
kare: kry kre karay
karein: karen kren karain karaein kryn
karne: krny karnay krne karny
karna: krna
kar: kr ker
karo: kro
karta: krta
karti: krti
hota: hta
hoti: hotai_x
kabhi: kabi kbhi kbi
abhi: abi abhe
sabhi: sabi
phir: fir phr phir
bhai: bai bhae bhaii
yaar: yar yr yrr yaar
please: plz pls plzz plez plzzz
you: u
and: nd
baat: bat baat
bas: bs bss
matlab: mtlb
message: msg
tak: tk
sath: saath sth
sirf: srif
sakte: skty sktay saktay
zyada: zyda zayada jyada ziada
acha: accha achcha achha achaa
achi: acchi achhi
ache: achy achay
theek: theak thik tek theak thek
dekh: daikh
dekha: daikha dekha
inshallah: insallah inshaallah inshalla
mashallah: mashaallah mashalla mashaalla mashallha
ameen: aameen ameeen amin aamen
allah: allha
alhamdulillah: alhamdhulilah alhamdulilah
wajah: waja
waqt: waqat
poora: pora pura
poori: pori
poore: pore pory
aurat: orat
awaz: awaaz
cheez: chez cheez
hukumat: hakomat hakumat hukomat
hamesha: hamaisha hamesha hmesha
behtar: bahter behter
shuru: shuro
kutte: kuttay kute
pehle: pehly phaly pahle pahley pheley pehlay
mohabbat: muhabat mohabat muhabbat
bahar: bahir
gaane: gane
kaash: kash
jate: jatay
khoobsurat: khubsorat khobsorat khubsurat
tujhe: tujhay tujhy
mujhe: mujhy mujhay muje mje mjhe mjy mujhai
hamein: hamain humein hame
hamare: hamaray hmary hamary humare
hamari: hmari humari
tumhare: tumharay
tumhein: tumhay
unhein: unhay unhe
inhein: inhe
inhon: inho
jinhon: jinho
mere: meray
tere: tery
itne: itny itnay intne
kitne: kitnay
saare: sary saray sarey sare
saara: sara_x
kehna: kahna
kehte: kehty kehtay
kehta: kahta
kehti: kahti
wale: walay waly
sahab: saab sahib
jaan: jan_ur
bare: bary baray
bara: bada
chahiye: chaiye chahye chaye chahy chahie chahiay
ghatiya: ghateya
lena: laina
gaana: gana
kaafi: kafi
lanati: lanti
baaqi: baqi
paise: paisay paisy pese
lanat: lant lanat
jaye: jay jae jai
aati: ati
aate: aty aaty
aaya: aya
aaye: aye
aayi: aai
aao: ao
aana: ana_x
aane: anay
jagah: jaga jagha
wahan: waha
shukriya: shukariya shukria
mein: mien mei_x
thoda: thora
thodi: thori
warna: warana
sukoon: sakon
khandan: kahndan
kya: kia_x
kiya: keya kiya_x
kuch: kuchh kuch
upar: oper
kitna: kitna
tarah: tarha trha
lekin: lakin lkn
isne: isny
usne: usny
lage: lagay
karega: karyga
iske: isky
use: usy
sakon: sakon_x
bande: bandy
kijiye: kijeay kijiay
pyari: payari
pyar: payar pyaar
pyara: payara
drame: dramy dramay
subah: subha
sukoon: sakoon
khush: kush
kharab: khrab
zindagi: zindgi
duniya: dunia dunya
izzat: izat
insaan: insan
musalman: musalmaan
bewaqoof: bewakoof
hidayat: hedayat
zaleel: zalil
jahil: jahel
"""

# merged words that are split on Urdu word boundaries (v4 convention, as in v3)
SPLIT_MAP = {
    "maine": "main ne", "mene": "main ne", "mainay": "main ne", "meine": "main ne",
    "hogaya": "ho gaya", "hogya": "ho gaya", "hogayi": "ho gayi", "hogai": "ho gayi", "hogaye": "ho gaye",
    "hojaye": "ho jaye", "hojay": "ho jaye", "hojata": "ho jata", "hojati": "ho jati", "hojate": "ho jate",
    "horaha": "ho raha", "horahi": "ho rahi", "horahe": "ho rahe", "hoskta": "ho sakta", "hosakta": "ho sakta",
    "kardeya": "kar diya", "kardiya": "kar diya", "kardia": "kar diya", "kardi": "kar di", "kardo": "kar do",
    "kardain": "kar dein", "kardein": "kar dein", "kardain": "kar dein", "kardy": "kar de", "karde": "kar de",
    "karliya": "kar liya", "karlia": "kar liya", "karkay": "kar ke", "karke": "kar ke", "karky": "kar ke",
    "krke": "kar ke", "krdo": "kar do", "krdia": "kar diya", "krdiya": "kar diya", "krna": "karna",
    "sabsy": "sab se", "sabse": "sab se", "sabsay": "sab se", "isliye": "is liye", "islye": "is liye",
    "isliay": "is liye", "ismay": "is mein", "ismein": "is mein", "isme": "is mein", "usme": "us mein",
    "usmein": "us mein", "jisme": "jis mein", "sunky": "sun ke", "sunke": "sun ke", "dekhke": "dekh ke",
    "dekhky": "dekh ke", "jaake": "ja ke", "aake": "aa ke", "mujhse": "mujh se", "tumse": "tum se",
    "dekhkr": "dekh kar", "dekhekar": "dekh kar", "ajaonga": "aa jaunga", "ajaunga": "aa jaunga",
    "ajaongi": "aa jaungi", "ajaungi": "aa jaungi", "karonga": "karunga", "karongi": "karungi",
    "kardeta": "kar deta", "kardeti": "kar deti", "kardete": "kar dete", "haiap": "hai aap",
    "haiapko": "hai aapko", "hyapko": "hai aapko", "jinhone": "jinhon ne", "unhone": "unhon ne",
    "inhone": "inhon ne", "khakar": "kha kar", "jakar": "ja kar", "akar": "aa kar",
    "humse": "hum se", "aplog": "aap log", "ispay": "is pe", "ismein": "is mein",
}

# tokens kept exactly as written (laughter, interjections, Punjabi markers, abbreviations)
KEEP = set("""haha hahaha hahahaha hahahahaha hehe hehehe lol lmao omg wow uff ufff oye oh ahh aww hmm hmmm
nu da di de te tusi sanu menu tenu vich kithe ae bc mc pti pmln ppp pdm mqm ispr isi ary geo dr mr mrs s a w saw
pbuh rs ok""".split())

# laughter / onomatopoeia patterns are matched in build_v4.py with a regex as well

# ------------------------------------------------------------------ context word sets
CONJ_START = set("""kash kaash aur or lekin magar to tou ke k ky kay keh kyunki kyun jab agar phir fir ab bhi jo ya balke warna
isliye tab yaar yar bhai ji jee haan han nahi bas sirf aaj kal abhi""".split())
FIRST_PERSON_NEXT = set("""ne nay ny to tou bhi khud hamesha roz kabhi apni apne apna tumhe tumhein aapko apko aap
ap tum isko usko ise use is us yeh ye yai woh wo kal aaj abhi bilkul sirf bas hoon hun hu tha thi karunga
karungi jaunga jaungi chahta chahti sochta sochti samajhta samajhti""".split())
VERB_AUX = set("""hota hoti hote raha rahi rahe gaya gayi gaye sakta sakti sakte chahiye hua hui hue karta karti
karte jata jati jate diya liya kiya mila mili mile laga lagi lage lagta lagti lagte aata aati aate tha thi the
chuka chuki chuke deta deti dete leta leti lete rakha rakhi karna karni karne hona honi hone jana jani jane
dena deni dene lena leni lene banana banani aana aani rehna rehni kehna kehni dekhna dekhni sunna sunni
samajhna bolna bolni likhna likhni""".split())
HI_PREV = set("""bohat bohot bht bhot bahut buhat itna itni itne kitna kitni yeh ye yai woh wo wahi yahi ab abhi tab aise waise ek sirf aap ap tum main hum mujhe usi isi aaj
kal jab apni apne apna mera meri mere""".split())
NEW_CLAUSE = set("""ke k ky kay keh aur or lekin magar jo to tou kyunki na ya yar yaar bhai sir ji jab agar phir ab
is us kya aap ap tum main hum yeh ye yai woh wo bhi""".split())
CHHOR_NEXT = set("""ky kay kr ker dy dia dya deya dena do diya di de dein dena dete deta deti kar ke k gaya gaye gayi rahe raha rahi chuka chuke chuki
dain dy""".split())
GAYE_PREV = set("ho chale chaly aa ja reh so mar bhaag ban mil".split())
BAAD_PREV = set("ke k ky kay is us iske uske jis saal din mahine hafte arse".split())

# English spellings that are also very common Roman Urdu words: never treated as English evidence
URDU_HOMOGRAPHS = set("""use be india indian pakistan to is in us me he main or the bad hi ho so do par pe sun mar are bat ban hum tab may mein
ye yeh na man men din bus pay day say key both bar han ki ka ko se ne aa ab sab i a no hai jab kar kam
dil sir ji wah bhai log""".split())
# English loanwords used constantly inside Urdu sentences: tagged EN, but not evidence of an English sentence
LOANS = set("time name song show film drama serial story family team point last number video episode".split())
URDU_HOMOGRAPHS |= LOANS

# frequent words of the real comments that are already in an acceptable spelling (reviewed; kept as written)
SOCIAL_OK = """
"""

# 'he' directly before an auxiliary is the emphatic ہی (daku he hongy -> daku hi honge)
AUX_AFTER_HI = set("""hai hain tha thi the hoga hogi honge hongy hongay ho hota hoti hote rehta rehti rehte rahe raha
rahi karta karti karte""".split())

# English words after which me / may / men are English (for me, to me, love me, ...)
EN_BEFORE_ME = set("for to with let give love call tell help miss like and about kill told show".split())
# kiya (did) after a 'ne' subject: main ne ... kiya hai / kiya hua tha
AFTER_KIYA = set("hai tha thi the hain hua huwa hui hue gaya gayi gaye jata jaye".split())
# mare -> mere (my) before these words; otherwise mare (killed / hit) is kept
AFTER_MERE = set("""sath saath pas paas liye bhai behen dil ghar papa abbu ammi khayal khyal dost pyare pyaray
sath samne bete beti bachay bache husband shohar""".split())

# 'he' before a possessive is the copula ہے (yeh he mera ghar)
POSSESSIVE = set("mera meri mere hamara hamari hamare tumhara tumhari tumhare aapka aapki aapke apka apki uska uski unka unki iska iski".split())

KAISE_NEXT = set("""ho hai hain hy h he kar kr karte karti karun karein karen kren hua hui hue huwa hoga hogi honge
bataun samjhaun lagta lagti lage pata mumkin kaha kahe hota hoti hote rahe raha rahi sakta sakti sakte bhool""".split())
MARE_NEXT = set("""gaye gaya gayi gay jaye jayen ga ge gi jate jati jata hue huay the thay tha jaenge jayenge""".split())
NE_WORDS = set("ne ny nay mene maine mainy meny mny tumne humne usne isne unhone inhone aapne kisne jisne".split())
