# 모듈형 위젯 아키텍처

이 프로젝트는 LLM 기반 체크리스트 생성 및 과제 평가 시스템을 위한 모듈형 위젯 아키텍처를 제공합니다.

## 주요 기능

- **체크리스트 생성**: 수행평가 정보를 입력하여 평가 기준 체크리스트 자동 생성
- **과제 평가**: 학생 과제를 평가 기준에 따라 평가하고 결과 시각화

## 아키텍처 구조

이 시스템은 모듈형 아키텍처로 구현되어 있으며, 다음과 같은 주요 구성 요소를 포함합니다:

### 기본 클래스

- **BaseWidgetManager**: 모든 위젯 매니저의 기본 클래스
- **BaseComponent**: 모든 구성 요소의 기본 클래스

### 구성 요소

- **InputWidgetsComponent**: 기본 정보(과목, 제목, 유형 등) 입력을 위한 위젯
- **TemplateWidgetsComponent**: 시스템 지시와 프롬프트 템플릿 편집을 위한 위젯
- **ModelSelectionComponent**: LLM 모델 선택 및 파라미터 설정을 위한 위젯
- **OutputComponent**: 결과 표시 및 저장을 위한 위젯
- **SubmissionWidgetsComponent**: 학생 과제 제출 및 파일 업로드를 위한 위젯
- **ChecklistComponent**: 체크리스트 선택 및 관리를 위한 위젯

### 특화 매니저 클래스

- **ChecklistCreationManager**: 체크리스트 생성 기능을 위한 매니저
- **AssignmentEvaluationManager**: 과제 평가 기능을 위한 매니저

## 사용 방법

### 체크리스트 생성 위젯 사용

```python
from widgets_core import ChecklistCreationManager

# 체크리스트 생성 매니저 생성
checklist_manager = ChecklistCreationManager()

# 위젯 표시
checklist_manager.display_all()
```

### 과제 평가 위젯 사용

```python
from widgets_core import AssignmentEvaluationManager

# 과제 평가 매니저 생성
evaluation_manager = AssignmentEvaluationManager()

# 위젯 표시
evaluation_manager.display_all()
```

## 커스텀 위젯 구성

기존 구성 요소를 조합하여 새로운 위젯 인터페이스를 쉽게 만들 수 있습니다:

```python
from widgets_core import BaseWidgetManager, InputWidgetsComponent, ModelSelectionComponent

# 커스텀 위젯 매니저 생성
class SimpleWidgetManager(BaseWidgetManager):
    def __init__(self):
        super().__init__()
        
        # 필요한 컴포넌트만 추가
        input_component = InputWidgetsComponent(self, include_grade=False)
        model_component = ModelSelectionComponent(self)
        
        # 컴포넌트 등록
        self.add_component('input', input_component)
        self.add_component('model', model_component)
    
    def display_all(self):
        # 간단한 레이아웃 구성 및 표시
        # ...
```

## 장점

- **재사용성**: 구성 요소는 여러 작업에서 재사용 가능
- **유지보수성**: 구성 요소 기능 변경은 한 곳에서만 이루어짐
- **유연성**: 필요에 따라 새로운 구성 요소 쉽게 추가 가능
- **명확성**: 관심사 분리로 코드 이해 용이
- **확장성**: 기존 구성 요소를 조합하여 새로운 작업별 매니저 쉽게 생성

## 실행 방법

Jupyter 노트북에서 예제 파일을 실행하여 위젯을 사용할 수 있습니다:

```bash
jupyter notebook widget_example.ipynb
``` 