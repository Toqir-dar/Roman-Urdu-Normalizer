"""Type-level review of the real social comments in dataset.csv (input to build_v4.py).

Written from annotation/review_unknown.tsv (regenerate with `python build_v4.py review`) and review_types.tsv (every word type seen >= 5 times that the
normalizer did not already know). Reviewer: Claude (AI) -- spot-check before publishing.
"""

# variant>canonical ('+' = space: merged words are split on Urdu word boundaries)
SOCIAL_MAP_2 = """
py>pe iss>is mary>mere yah>yeh kyu>kyun kra>kara rakhy>rakhe bachy>bache choro>chhoro puri>poori bta>bata olad>aulad
rakhay>rakhe zindah>zinda maf>maaf zulam>zulm chu>chhu kabool>qabool paish>pesh khanay>khane jeya>jiya
allahapko>allah+aapko jhota>jhoota subko>sab+ko agay>aage doran>dauran agae>aage pechay>peeche quom>qaum khail>khel
ami>ammi dosri>doosri nasal>nasl himat>himmat jhoti>jhooti hazar>hazaar bahen>behen krwa>karwa farak>farq
farmay>farmaye qaiam>qaim mout>maut dill>dil bachay>bache bhen>behen rooti>roti abu>abbu masom>masoom
istaimal>istemal jhoty>jhoote shksiyat>shakhsiyat uss>us pae>pe bkwas>bakwas batain>baatein beth>baith banay>bane
bap>baap mily>mile murshad>murshid oqaat>auqaat darkhuwast>darkhwast g>ji pahaley>pehle baijh>bhej marnay>marne
darama>drama ilam>ilm inna>itna showbaz>showbiz khaton>khatoon jisay>jise bd>baad begerat>begairat jany>jane
rahna>rehna lagtay>lagte sakhat>sakht saky>sake vah>wah daynay>dene chahty>chahte psnd>pasand milay>mile acche>ache
kahni>kehni khinzer>khinzir rastay>raste lagy>lage yrrr>yaar mt>mat pahile>pehle zaheni>zehni letay>lete lac>lakh
roj>roz bolty>bolte nhn>nahi daty>dete lrki>larki naani>nani tumhe>tumhein batein>baatein qoum>qaum aby>abe
tarbiat>tarbiyat galiya>galiyan kamiyab>kamyab laanti>lanati asman>aasman faisala>faisla hogae>ho+gaye
hoqoq>huqooq khared>kharid gajab>ghazab tmhy>tumhe chaly>chale muh>munh je>ji makam>maqam kapray>kapre
parhnay>parhne osm>awesome isay>ise pheli>pehli qk>kyunki inky>inke chezon>cheezon darma>drama fittay>fitte
hongy>honge betha>baitha jaisy>jaise aankon>aankhon bund>band zikar>zikr zimadari>zimmedari shaks>shakhs
samny>samne kahzana>khazana sadqay>sadqe ghatia>ghatiya poch>pooch lagae>lagaye dosra>doosra arsay>arse almi>aalmi
larkiya>larkiyan gee>ji ju>jo taiz>tez gandy>gande rakhain>rakhein hii>hi dusra>doosra inhain>inhein maanay>maane
asani>aasani bajy>baje haii>hai rabita>rabta dosron>doosron leay>liye dosre>doosre bot>bohat bahtreen>behtareen
hukamrano>hukmranon huwy>hue kion>kyun dakhtay>dekhte shikasht>shikast choo>chhu choty>chhote kii>ki
ayega>aayega mgr>magar astagfirullah>astaghfirullah insano>insanon vi>bhi mujay>mujhe tbhi>tabhi amen>ameen
smj>samajh dusri>doosri unhy>unhein dalo>daalo sb>sab marr>mar waloon>walon kisam>qisam sunkar>sun+kar
begherat>begairat pakapko>pak+aapko moun>munh tmhen>tumhein jeetnay>jeetne nikly>nikle kisy>kisi mashjid>masjid
zbrdst>zabardast hea>hai banai>banayi uskay>uske moka>mauqa guzishta>guzashta roo>ro begarat>begairat
pyary>pyare balkay>balke larkay>larke tayaar>tayyar hukam>hukm kio>kyun nahein>nahi samait>samet achnk>achanak
jantay>jante pahichan>pehchan jarurat>zaroorat qemat>qeemat bhn>behen chezain>cheezein oy>oye aglay>agle
bass>bas q>kyun hia>hai ehsas>ehsaas ahsas>ehsaas larkiyo>larkiyon chorail>churail eik>ek nazer>nazar
unhen>unhein mohobbat>mohabbat krain>karein alami>aalmi baten>baatein bhul>bhool khoty>khote bcha>bacha
dilo>dilon duao>duaon jayain>jayein wqt>waqt ayga>aayega kutto>kutton hawaly>hawale hamri>hamari gaein>gayin
dii>di barey>bare hume>hamein pohanch>pahunch hmain>hamein atay>aate qom>qaum bnai>banayi sae>se howay>hue
dyn>dein qoom>qaum muj>mujh ayi>aayi laken>lekin bekarar>beqarar wada>waada kharich>kharij andr>andar
hamen>hamein karae>karaye aacha>acha chye>chahiye pakistanio>pakistaniyon agai>aa+gayi saaminay>saamne rhaa>raha
hadd>had oar>aur mainy>main+ne kamiyabi>kamyabi wesa>waisa bhol>bhool hakoomat>hukumat bolny>bolne glt>galat
gareebo>gareebon mot>maut ankh>aankh bethy>baithe mashahoor>mashhoor roak>rok bnane>banane sacchi>sachi
chiz>cheez bagarat>begairat jakar>ja+kar saiasat>siyasat zalemon>zalimon vedio>video huey>hue piche>peeche
lainay>lene frmay>farmaye jye>jaye pocha>poocha mahinay>mahine nojawan>naujawan zaberdast>zabardast danay>dane
majbor>majboor haina>hai+na nii>nahi kyunky>kyunki ajate>aa+jate humy>hamein dalna>daalna nechy>neeche
musalat>musallat fakhar>fakhr qanon>qanoon aajay>aa+jaye ajao>aa+jao nyc>nice begharat>begairat khouf>khauf
liken>lekin bcz>because kyuki>kyunki walda>walida rhay>rahe lelo>le+lo bachny>bachne hmy>hamein maamly>maamle
shaor>shaoor humen>hamein kse>kaise jabardast>zabardast mulak>mulk naimat>nemat bhtr>behtar insha'allah>inshallah
nhii>nahi mahfoz>mehfooz kesi>kaisi intaqal>inteqal chay>chai peche>peeche ibtadae>ibtedai degar>deegar ghlt>galat
yhe>yahi khotay>khote subky>sab+ke anty>aunty ganay>gaane munafik>munafiq gey>ge mna>mana ulad>aulad
ortain>aurtein besharm>besharam pey>pe mujha>mujhe nahen>nahi ane>aane bri>bari araha>aa+raha tumari>tumhari
kaamo>kaamon mertaba>martaba jatey>jate apse>aap+se sunae>sunaye soo>so haftay>hafte cheni>cheeni frmaye>farmaye
bnana>banana pohch>pahunch musalmano>musalmanon ankhen>aankhen chuky>chuke sadigi>sadgi allaha>allah
kisnay>kisne kom>qaum molana>maulana rizaq>rizq smjhi>samjhi howae>hue kareeb>qareeb hamay>hamein koch>kuch
lotay>lote dehan>dhyan rho>raho kero>karo pehley>pehle kiu>kyun mea>mein bsss>bas ulaad>aulad chahra>chehra
mazy>maze tuo>to medam>madam btana>batana bechary>bechare kum>kam logu>logon kroge>karoge intehae>intehai
bnaty>banate mehsos>mehsoos btaya>bataya dou>do pagl>pagal ajata>aa+jata hojaiga>ho+jayega awr>aur hamy>hamein
jayda>zyada chahiya>chahiye arha>aa+raha aamin>ameen tankeed>tanqeed allahapki>allah+aapki zamany>zamane
ansu>aansu asay>aise labaik>labbaik foran>fauran siriyal>serial aunti>aunty mza>maza hazaro>hazaron neyat>niyat
faisley>faisle btaye>bataye khus>khush agarchay>agarche hoty>hote bhary>bhare aagya>aa+gaya humary>hamare
pocho>poocho toufeq>taufeeq krenge>karenge dekhny>dekhne agya>aa+gaya jaruri>zaroori takriban>taqreeban
boly>bole bawajod>bawajood ess>is aplogo>aap+logon banaty>banate banany>banane lanut>lanat haia>hai chaty>chahte
aapse>aap+se vali>wali haqiqe>haqiqi bhj>bhej bhan>behen giya>gaya arahi>aa+rahi suntay>sunte krn>karein
etna>itna pury>poore ikhteyar>ikhtiyar jehaad>jihad ahtijaj>ehtijaj dekhkar>dekh+kar jayada>zyada
jinhain>jinhein waqeel>wakeel farmaey>farmaye kataar>qataar jindgi>zindagi angraizi>angrezi awaj>awaz
masaallah>mashallah natijay>natije tamana>tamanna bivi>biwi betay>bete ham>hum gus>ghus jesay>jaise bnaye>banaye
chaheye>chahiye kroo>karo sedha>seedha aeham>aham pyr>pyar masly>masle ensy>inse apney>apne griftar>giraftar
kerte>karte gariftar>giraftar likhtay>likhte muja>mujhe sub>sab hojao>ho+jao wao>wow tawaja>tawajjo
pehnay>pehne zindabaad>zindabad ahmeyat>ahmiyat wasol>wasool mahool>mahaul waqaf>waqf lgany>lagane araz>arz
hm>hum mer>mere rhta>rehta dekhy>dekhe unkay>unke bemar>bimar parey>pare tuj>tujh dosry>doosre shobay>shobe
apneapko>apne+aap+ko hamesa>hamesha kufar>kufr mubarik>mubarak aja>aa+ja aurto>aurton sidha>seedha
seriel>serial bety>bete seht>sehat nmbr>number ajkal>aaj+kal mokammal>mukammal krein>karein yhan>yahan
bnany>banane chotiya>chutiya tarhan>tarah meherbaani>meharbani ahteram>ehteram waqeya>waqia nahii>nahi
lotta>lota daramy>drame bato>baaton awsome>awesome dramo>dramon kardena>kar+dena buhut>bohat bagerat>begairat
zindge>zindagi mukamal>mukammal mainapko>main+aapko ronay>rone deye>diye rabb>rab jhooth>jhoot kernay>karne
khty>kehte palko>palkon rozay>roze enhoney>inhon+ne mahobbat>mohabbat frmaey>farmaye vale>wale bating>batting
sajdah>sajda duaa>dua mahrat>maharat colum>column vajah>wajah shauk>shauq kabar>qabr prta>parta
chaltay>chalte chaheay>chahiye phn>phone joh>jo rakhtay>rakhte dolat>daulat beshrm>besharam ankho>aankhon
pyaari>pyari moat>maut agr>agar kurbani>qurbani adhora>adhoora gurbat>ghurbat lambay>lambe kameeny>kameene
dkh>dekh hoa>hua rehai>rihai bje>baje boltay>bolte bigair>baghair farmye>farmaye reheem>raheem matti>mitti
trhan>tarah dewana>deewana qabal>qabil jeetay>jeete dkho>dekho naii>nahi bethe>baithe muqam>maqam
tarika>tareeqa pohanchani>pahunchani khobi>khoobi chata>chahta bachion>bachiyon haiapko>hai+aapko
dawae>dawai beshk>beshak aay>aaye batmeez>badtameez waloo>walon selaab>sailab likhay>likhe neechay>neeche
keray>kare phli>pehli sabki>sab+ki chukay>chuke sall>saal krde>kar+de harktain>harkatein lga>laga
meny>main+ne tumare>tumhare tmhre>tumhare medan>maidan pichy>peeche lekr>le+kar badiya>badhiya peechy>peeche
hamderdi>hamdardi zrurt>zaroorat tym>time chlega>chalega yae>yeh knsa>kaunsa karahai>karahi shayeri>shayari
masroofyat>masroofiyat fermay>farmaye rhna>rehna leaga>lega bchy>bache nawazay>nawaze chz>cheez uspe>us+pe
rakhnay>rakhne zidd>zid baatain>baatein doosray>doosre aapane>aap+ne aache>ache begert>begairat naheen>nahi
krva>karwa behj>bhej becharay>bechare bohtt>bohat shru>shuru ussy>use loty>lote idarey>idare mja>maza
mangwaliya>mangwa+liya rkhay>rakhe sayari>shayari tjhy>tujhe arhi>aa+rahi uthae>uthaye neapko>ne+aapko
tuje>tujhe zimadar>zimmedar ghareebo>ghareebon dubara>dobara betiyo>betiyon bolay>bole akhiri>aakhri khub>khoob
thoray>thore paigam>paigham martabah>martaba fer>phir nambar>number ly>le drma>drama baquaida>baqaida
begair>baghair muu>munh shyd>shayad nokar>naukar istmal>istemal chorna>chhorna lagra>lag+raha nokri>naukri
hazri>haazri chorni>chhorni vedios>videos bny>bane baryani>biryani shaheedo>shaheedon farmy>farmaye ayge>aayega
mosiqi>mausiqi jarur>zaroor kabil>qabil bannay>banne dob>doob andhy>andhe bhtt>bohat hmen>hamein pochta>poochta
tashadud>tashaddud waqayat>waqiat tarike>tareeqe es>is lae>laaye nekal>nikal kehnay>kehne ullo>ullu mooh>munh
milty>milte yehe>yahi orten>aurtein mulq>mulk btata>batata zror>zaroor promot>promote bhekari>bhikari
chahia>chahiye duwa>dua dea>diya ganjay>ganje dramma>drama merii>meri humny>humne peh>pe kharedi>kharidi
lanatain>lanatein daiko>dekho bahany>bahane pochna>poochna meyat>mayyat tre>tere bich>beech wlo>walon
slamt>salamat skta>sakta khuch>kuch koye>koi durr>door bhttt>bohat bnty>bante ayee>aayi uta>utha daadi>dadi
hanji>haan+ji khuwab>khwab laikin>lekin kiaa>kya kharchay>kharche vadiya>badhiya vadia>badhiya fitay>fitte
bemisal>bemisaal awaam>awam kamiyabiyon>kamyabiyon moqablon>muqablon gilazat>ghilazat khahish>khwahish
tajrba>tajurba nachnay>nachne karaingay>karenge fasila>faasla missal>misaal bda>bara ahista>aahista
bakhobi>bakhoobi zabrdst>zabardast lyn>lein jindagi>zindagi
skti>sakti jaey>jaye duayn>duaen inn>in ples>please kml>kamal deay>diye dye>diye nizame>nizam allh>allah
zeada>zyada hmry>hamare hafzat>hifazat bouht>bohat bhaut>bohat whaan>wahan tume>tumhe kerti>karti releted>related
ajaty>aa+jate koty>kutte gyn>ge kiyhe>kiya+hai laita>leta luty>loote arubo>arbon
zaida>zyada kown>kaun movi>movie lagya>lagaya
kaay>ke daay>de naay>ne jada>zyada chayee>chahiye mi>mein dhooka>dhoka suru>shuru isly>is+liye tmy>tumhe mny>main+ne hva>hua nahien>nahi ak>ek kel>kal
gud>good msla>masla behin>behen hien>hain kyoki>kyunki dekhkr>dekh+kar dosray>doosre kery>kare
jinhone>jinhon+ne rhii>rahi etni>itni atni>itni mohabbt>mohabbat pochy>pooche pochay>pooche
cahiye>chahiye chaie>chahiye chahiy>chahiye chyee>chahiye chutya>chutiya chutyaas>chutiya chutyas>chutiya hazart>hazrat hazarat>hazrat
mayoos>mayus baccho>bachchon lanet>lanat lante>lanatein dny>dein baato>baaton baaten>baatein
pakistaniyo>pakistaniyon pakistanion>pakistaniyon darkhawast>darkhwast beghairt>beghairat yahodi>yahoodi
skoon>sukoon ghalat>galat bnate>banate logn>logon lgte>lagte kyaa>kya aour>aur ankhein>aankhen
bikhari>bhikari zayda>zyada zahni>zehni peyary>pyare oncha>ooncha ghunda>gunda aacche>ache aacchi>achi
aaccha>acha asslam>assalam aslam>assalam alaekum>alaikum hamry>hamare ajaonga>aa+jaunga
ajaunga>aa+jaunga bapas>wapas taran>tarah mjah>mujhe luay>liye olu>ullu pathy>putte wajod>wajood
haiin>hain bhii>bhi apnee>apne kaas>kaash deakh>dekh kuto>kutton goliyan>goliyan darpaish>darpesh
chuta>chhuta umra>umrah ulma>ulama pazer>pazeer adakara>adakaara baich>bech
souna>sona chia>chahiye mozahra>muzahira khakar>kha+kar pakhton>pakhtun kachray>kachre joty>joote
ghomti>ghoomti armaan>arman wastay>wastey sochnay>sochne ispay>is+pe batt>baat
kitani>kitni gjb>ghazab ldki>larki qadyani>qadiani jaloos>juloos hajji>haji
delaata>dilata dupata>dupatta batai>batayi shadio>shadiyon khofnak>khaufnak
chur>chor imkan>imkaan ferma>farma pda>pada ensan>insaan motabiq>mutabiq aitraaf>aitraf
bethay>baithe akar>aa+kar nei>nahi draz>daraaz kardeta>kar+deta yesa>aisa gahr>ghar
banya>banaya siwae>siwaye thankyou>thank+you namona>namuna behno>behenon dosro>doosron
howa>hua bheno>behenon aapni>apni sbi>sabhi luta>loota gie>gayi wagera>wagaira
humse>hum+se mafad>mufaad kripya>kripaa docter>doctor nezdiqe>nazdeek jooth>jhoot
khti>kehte chwal>chawal bqi>baaqi hme>hamein zrort>zaroorat tbi>tabhi namonay>namune
chizen>cheezein aplog>aap+log bachan>bachchan parne>parhne kerty>karte btya>bataya
karwain>karwaein haiap>hai+aap thnx>thanks moive>movie diwnload>download talway>talwe
lagni>lagani jila>zilla naizi>niazi leeg>league toa>to gour>ghour mro>maro neh>nahi
maroo>maro qanone>qanooni walu>walon nishaniya>nishaniyan sapooto>sapooton
chodny>chhodne dedar>deedar
"""

ENGLISH_MORE = """aunty scene job shame turkish ost product face send bank suit dm pic background corrupt dj eyes car waste set
actors smile low issue poetry wife performance husband service salute shows year bomb colour singing request cheap
years mother makeup tell different imported kindly cover runs dress experience means fit couple stuff rangers future
shirts joke win month thin appreciate mom overacting stupid item politicians arrest words ma'am deserve buy apply
already superb fast brave refund dressing coca feeling roads shut read dislike riders months visit default
followers address muslims scenes salary establishment proper seriously notice actresses totally inbox publicity
economy screen comfortable chance channels badly fitting take looks leak eye milk welcome simple ending waiting sell
repeat hours choice exactly talented fraud filmy absolutely caption safe search supporter attitude pray loved place
fresh charges choosers unfollow remember innings knowledge marriage sheer possible sexy foreigner tight politician
religion hosting rape bye jobs countries roles etc bf leg since leaders showbiz prove products indians beach licence
dna prophet topic gf atleast batsman lets cricketers rich fabric clean lots expect value watching interested banned
rating ep loan unlike talking cherry blossom direction non called mail decent learn leave maam listen dialogue gov
illegal confidence popular believe ex left large female members fb cream put making ac missing mill worst pc
specially bluetooth population kidney voucher blast deserves due rights cantt handsome waist campaign fat families
attention qualification answer strap according admission sitting compare others rip earning own effort handlers
massage senate workers killer players promo honest saying overall taxes insult shit retirement lover drugs celebrity
react jealous soul disappointed idiot figure supporters village starting coat athlete terrorist skills allow interest
struggle guide fridge chill sr married trying nonsense offer logic response rest knows api vacancy pics hats budget
batting provide complaint till promote awesome nice because dogs missed ladies condition heard artists
category earn bullshit greatest worry useless puppet detail skip topics boring biggest fame boycott
mouth guy contract plans candidate constitution rules edit vulgarity healthy run father responsible
promoting continue fitness relationship exact fare easily adopt heat vouchers leather amount except
timing agency increase blessed mint trading concept collection uneducated what's selfie""".split()

NAMES_MORE = """foodpanda qalandars shabaz ubqari esakhelvi orton galib bushra musharaf raam attaullah zeus meera stallone
liaquat qureshi hera ertugrul zia sherry asim abid marilyn jouhar elahi minhas rajo munir tata butt russia firoz mishra
batman munawar keller prada shahrukh rida aftab munawwar husin monroe usa ricky hashir mohyeddin mohammed kohli depp
selma momal mehran roy bilquis kanwal uk diana omer virat ehsan uzmaa lagerlof wahaj valmiki mariyam johnny emma jolie
mathira munnawar musharf marium jullandhuri sarojini amrit chundrigar osman rfak samsung pandamart imtiaz""".split()

# frequent tokens too unclear to normalize: left as written, and rows containing them stay 'silver'
UNCLEAR = set("""y yn yy yyy yyyy yyyyy d pa p c m t l st ch j n ray eeeai ry ai rae du rd vp ic eh br vn bv pee ani
apk sum mal min shaa chaa bate kiss chore hoap dhakta rahiti rahai saisi lun pir mem hna doo phen suma summa ish thaak
lanta aho pasa kahinay janati rahinay sochitay samajhitay hifzo aaiyn chya baty ene batin bury jazay zana ghandh barr
pawa sunain kein janein jaige pahiche hau bae aw bo chee bais bachain sho rahu hoein dino lora anya putr twadi hor
dub yojna yojana zer shir chaya urat ia tul aahe rahet masher dhool koa bun saakhti oa dogi chuss chuu asania
laity aain tuh janatul lati goor khach ponch alhadigi dn pia kur aak bb hoto ve apus ey guzeer thu thuuu mon hundi
hou beh gunny wawa aus hoya hori bamsi raht bachoo oder naw ach gharat cs awla alaya""".split())

# SOCIAL_OK (frequent words already in an acceptable spelling) is generated below by build_v4.py's `review`
# command from the same review list: every reviewed type that is not mapped, English, a name or UNCLEAR.
SOCIAL_OK = """
mare bari ban hasil isko larki fatih harami zindabad halat chutiya unke admi hamara shair khaas sunta jeetna patwari
tamam daku gareeb afsoos jeo ramazan haawi raakhi unko rula jald kanjar chahe bina khawateen shamil inka begairat eman
chahati kirdar kaho beghairat ilawa chote kaano gaddar dimag yakeen parh molvi sabit qatal taraqi louta intakhab farig
banate akhir samne jang saalo sabar bibi ghali ilzam anjam jhoom qarar agaz faltu khoti khor bharta haqiqat anmool
indori gandu hongi chaat nakli karkerdagi gando kismat bechari gair utaar tumko jhum yay farmai usse karwao kapra
akhri nuksan quaid besharam mazak bolne tune kasoor hajj gaib bhikari badhiya bajhay jahilo naslo gairat subhanallah
taaluq chutiye maqboliyat dikhai pur tharki waldain pehen rahine badalte jag denge khelaryon bazi hamko iqraar khali
nasha rahega deni gyan tola billo shabash nizam barh shaitan shirk nikalte utna dalal khari izafa jisko loo behaya mou
utne barha aizaz adaab sawaab mulaqat dum riha bataye saalon maze didi gaal adbi pk abba pakra patwario kamino jumma
pml shifa muntakhib qayam khusra shakl jisne chukka tatti izhar barhi banayega sunder fav burai kameeni jore aapne kpk
jaani bala rone bhagwan karkun adalat payein janbahaq chupa haramkhor aoa chalega sharabi dikh mutmaeen fikr hansti
yaadgar mann bik wazeefa bheegi id pehene bhonk galti khushiyan sabiqa bhook tameer ay parda haty ainda mp mangwaayi
bechare thanku bataon shikayat laiq nigar padega lipstick jio com salahiyaton parhaya manta karobar lazmi ipl bhabhi
bsdk chutia jhagra bimari jazakallah murdabad soba taari sarko wahiyat jinke pakro behas chakkar amma tehrir ujala
humne lenge rahen pakray shaan sumameen nigari jiske barhe chanda unse likhni aitmad zehr lillahi phudi joota behayai
chehre hayat tik talluqaat sambhala moot zehar pehchana faqeer dheet bhale socho nazil manazar pareshani guftugo
khelni bikao allahallah naziren filhal awal yo cm garib jaib kg wasta tehat ghora dhandli jamhoriyat tumne jazak ashiq
mazi yaqeeni bindi bolu sawar madarchod aukaat dikhana khelte dhokha khabasat bean bnda mutassir galeez lathi bhenchod
parhaye jarihana karrawai milegi parhai gr tlp pila psl misaal uhday chaprasi taqreer majmooa daddu beshumaar jurrat
batate marte mujra fahashi seekho lolz mangne shree imdad bila najoomi talash qisam akhirat degi zalimi rajioon
shayaqeen lagani bator rukun lafafa dajjal nikalo badmash thapar chokhat ullu hartal raaton besharmi murgi khula
jaankari chuhe ishqiya namanzoor ps batane kapron kahun astaghfirullah aziyat teeno taza mehfooz assalam nali adha
yahoodi padta aankhen youthia sansad mangu musallat dekhein khusre chalak ganna natak kariye jalte putar wapasi shehar
ganne atma nizame angaiz awara dalte haqdar gg darjat btay ghori pehna sachai swt bhula khitaab khalistani patha
pazeerai dekhu ullah kanjri kameez wahin barqarar sehri wazarat rashan patwaran ptv sanatan sazish irada wafadar
lillahe kin jayein gharq tauba aksbandi infaradi parhe
"""
