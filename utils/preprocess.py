import kss
import re


def doc_preprocess(src_doc):
    '''
    Information
        문서(split 되지 않은 상태의 유튜브 스크립트 원본 text)를 넣었을 때
        온점 추가/제거 등의 전처리 수행 후 반환
        
    Arguments
        src_doc: input script
    '''
    # 소수점 제외한 온점은 모두 제거
    preprop = re.sub('([^\d])(\.)', lambda m: "{}".format(m.group(1)), src_doc)
    
    # kss 라이브러리를 통한 문장 분리
    preprop = kss.split_sentences(preprop)

    # 니다~ 인데 온점이 안붙는 경우 온점을 붙여줌 (규칙 추가 필요시 추가)
    preprop = re.sub('(니다)([^\.])', lambda m: "{}. ".format(m.group(1)), '. '.join(preprop))
    preprop = re.sub('(거죠)([^\.])', lambda m: "{}. ".format(m.group(1)), '. '.join(preprop))
    preprop = re.sub('(이죠)([^\.])', lambda m: "{}. ".format(m.group(1)), '. '.join(preprop))
    preprop = re.sub('(데요)([^\.])', lambda m: "{}. ".format(m.group(1)), '. '.join(preprop))
    
    # 온점 위치 기준(. )으로 문장 split -> \n(sep-token)을 붙여 최종 문서 반환
    fin = '\n'.join(preprop.split('. '))
    
    return fin
    
    
    