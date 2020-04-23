from django.shortcuts import render
from django.views.generic import TemplateView
from .forms import ImageForm
from .main import detect

# Create your views here.
class MyDetectorView(TemplateView):
    # 생성자
    def __init__(self):
        self.params = {'result_list':[],
                       'result_name':"",
                       'result_img':"",
                       'form': ImageForm()}

    # GET request (index.html 파일 초기 표시)
    def get(self, req):
        return render(req, 'mydetector/index.html', self.params)

    # POST request (index.html 파일에 결과 표시)
    def post(self, req):
        # POST 메서드에 의해서 전달되는 FORM DATA 
        form = ImageForm(req.POST, req.FILES)
        # FORM DATA 에러 체크
        if not form.is_valid():
            raise ValueForm('invalid form')
        # FORM DATA에서 이미지 파일 얻기
        image = form.cleaned_data['image']
        # 이미지 파일을 지정해서 얼굴 인식
        result = detect(image)

        # 얼굴 분류된 결과 저장
        self.params['result_list'], self.params['result_name'], self.params['result_img'] = result

        # 페이지에 화면 표시
        return render(req, 'mydetector/index.html', self.params)
