import io, sys
sys.path.insert(0, ".")
import pypdfium2 as pdfium
from src.azure_clients import get_azure_credential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

ep = "https://vision-pii-redaction.cognitiveservices.azure.com/"
pdf = pdfium.PdfDocument(r"samples\Goldman-Sachs-Pitchbook-Airvana.pdf")
img = pdf[0].render(scale=150/72).to_pil().convert("RGB")
buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
data = buf.getvalue()
c = ImageAnalysisClient(endpoint=ep, credential=get_azure_credential())

feats = [VisualFeatures.OBJECTS, VisualFeatures.TAGS, VisualFeatures.CAPTION,
         VisualFeatures.DENSE_CAPTIONS, VisualFeatures.READ, VisualFeatures.SMART_CROPS,
         VisualFeatures.PEOPLE]
for f in feats:
    try:
        r = c.analyze(image_data=data, visual_features=[f])
        print("SUPPORTED     ", f)
        if f == VisualFeatures.OBJECTS and r.objects:
            for o in r.objects.list:
                print("      obj:", o.tags[0].name, round(o.tags[0].confidence, 2), o.bounding_box)
        if f == VisualFeatures.TAGS and r.tags:
            print("      tags:", [(t.name, round(t.confidence, 2)) for t in r.tags.list][:15])
    except Exception as e:
        print("NOT SUPPORTED ", f, "->", str(e).splitlines()[0][:80])
