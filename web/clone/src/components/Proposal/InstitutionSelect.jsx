import React, { useState } from 'react';
import styled from 'styled-components';

const FormWrapper = styled.div`
  width: 100%;
`;

const SubTitle = styled.header`
  margin-bottom: 20px;
  h3 {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--gray90);
  }
`;

const StepIndicator = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 40px;
  border-bottom: 1px solid var(--gray20);
  padding-bottom: 20px;

  .order_item {
    display: flex;
    align-items: center;
    margin: 0 20px;
    color: var(--gray40);
    font-size: 1.6rem;
    
    em {
      display: inline-block;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background-color: var(--gray40);
      color: white;
      text-align: center;
      line-height: 24px;
      font-size: 1.2rem;
      margin-right: 8px;
      font-style: normal;
    }

    &.on {
      color: var(--primary50);
      font-weight: 700;
      
      em {
        background-color: var(--primary50);
      }
    }
    
    &.completed {
        color: var(--gray80);
        em {
            background-color: var(--gray60);
        }
    }
  }
`;

const InfoBox = styled.div`
  background-color: var(--gray5);
  padding: 20px;
  border-radius: var(--radius-sm);
  margin-bottom: 30px;
  
  .box_pt {
    p {
      font-size: 1.4rem;
      color: var(--gray80);
      margin-bottom: 10px;
      
      strong {
        font-weight: 700;
      }
    }
    
    ul {
      list-style: disc;
      padding-left: 20px;
      
      li {
        font-size: 1.3rem;
        color: var(--gray70);
        margin-bottom: 5px;
        
        a.red {
            color: var(--danger50);
            font-weight: 700;
            text-decoration: underline;
        }
      }
    }
  }
`;

const SectionTitle = styled.div`
  margin-bottom: 15px;
  h4 {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--gray90);
    border-left: 4px solid var(--primary50);
    padding-left: 10px;
  }
`;

const AutoRecommendationBox = styled.div`
  margin-bottom: 40px;
  
  .kigwan_sel {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 10px;
    margin-bottom: 10px;
    
    button {
      width: 100%;
      padding: 12px;
      background-color: white;
      border: 1px solid var(--gray30);
      border-radius: var(--radius-xs);
      color: var(--gray80);
      font-size: 1.4rem;
      cursor: pointer;
      transition: all 0.2s;
      
      &:hover {
        border-color: var(--primary50);
        color: var(--primary50);
        background-color: var(--primary5);
      }
    }
  }
  
  .important_txt {
    font-size: 1.3rem;
    color: var(--gray60);
  }
`;

const ManualSelectionBox = styled.div`
  border: 1px solid var(--gray20);
  border-radius: var(--radius-sm);
  overflow: hidden;
  
  .slideBtn {
    display: block;
    padding: 15px;
    background-color: var(--gray5);
    color: var(--gray80);
    font-weight: 600;
    text-align: right;
    font-size: 1.3rem;
    border-bottom: 1px solid var(--gray20);
    
    &:after {
        content: ' ▲';
        font-size: 1rem;
    }
    
    &.closed:after {
        content: ' ▼';
    }
  }
  
  .auto_box {
    padding: 20px;
    
    .value_smallbox {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 20px;
        align-items: center;
        
        strong {
            font-weight: 700;
            margin-right: 10px;
        }
        
        .radioBtn {
            display: flex;
            align-items: center;
            font-size: 1.4rem;
            
            input {
                margin-right: 5px;
            }
            
            label {
                cursor: pointer;
            }
        }
    }
    
    .schBox {
        background-color: var(--gray5);
        padding: 15px;
        margin-bottom: 20px;
        border-radius: var(--radius-xs);
        
        .schRow {
            display: flex;
            align-items: center;
            
            label {
                margin-right: 10px;
                font-weight: 600;
            }
            
            .sch_btn_right {
                display: flex;
                gap: 5px;
                
                input {
                    height: 36px;
                    border: 1px solid var(--gray30);
                    padding: 0 10px;
                    width: 200px;
                }
                
                button {
                    height: 36px;
                    padding: 0 15px;
                    background-color: var(--gray80);
                    color: white;
                    border: none;
                    border-radius: var(--radius-xs);
                    cursor: pointer;
                }
            }
        }
    }
    
    .inner_bd {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 10px;
        
        button {
            width: 100%;
            padding: 10px;
            background-color: white;
            border: 1px solid var(--gray30);
            border-radius: var(--radius-xs);
            color: var(--gray80);
            font-size: 1.3rem;
            cursor: pointer;
            text-align: left;
            
            &:hover, &.on {
                border-color: var(--primary50);
                color: var(--primary50);
                background-color: var(--primary5);
            }
        }
    }
    
    .auto_more {
        text-align: center;
        margin-top: 20px;
        
        button {
            padding: 8px 20px;
            border: 1px solid var(--gray30);
            background-color: white;
            border-radius: 20px;
            cursor: pointer;
            font-size: 1.3rem;
            
            &:hover {
                background-color: var(--gray5);
            }
        }
    }
  }
`;

const ConfirmationBox = styled.div`
  margin-top: 30px;
  padding: 30px;
  background-color: var(--primary5);
  border: 1px solid var(--primary20);
  border-radius: var(--radius-sm);
  text-align: center;
  font-size: 1.8rem;
  color: var(--gray90);
  
  strong {
    font-weight: 700;
  }
  
  span {
    color: var(--primary60);
    font-weight: 700;
  }
`;

const ButtonArea = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid var(--gray20);

  .btnA_l, .btnA_r {
    display: flex;
    gap: 10px;
  }

  button {
    height: 45px;
    padding: 0 20px;
    font-size: 1.5rem;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 500;

    &.gray {
      background-color: var(--gray10);
      border: 1px solid var(--gray30);
      color: var(--gray80);
      
      &:hover {
        background-color: var(--gray20);
      }
    }

    &.line {
      background-color: white;
      border: 1px solid var(--primary50);
      color: var(--primary50);
      
      &:hover {
        background-color: var(--primary5);
      }
    }

    &.fill {
      background-color: var(--primary50);
      border: 1px solid var(--primary50);
      color: white;
      
      &:hover {
        background-color: var(--primary60);
      }
    }
  }
`;

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
`;

const ModalContent = styled.div`
  background-color: white;
  width: 600px;
  border-radius: var(--radius-md);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  overflow: hidden;
  
  .layerPop_top {
    background-color: var(--primary50);
    color: white;
    padding: 15px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    
    strong {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    button {
        background: none;
        border: none;
        color: white;
        cursor: pointer;
        font-size: 1.4rem;
    }
  }
  
  .def_lPop_body {
    padding: 30px;
    
    p {
        margin-bottom: 20px;
        color: var(--gray70);
        font-size: 1.4rem;
    }
    
    .titBox {
        margin-bottom: 15px;
        strong {
            font-size: 1.8rem;
            color: var(--gray90);
        }
    }
    
    .preview {
        border-top: 1px solid var(--gray80);
        margin-bottom: 20px;
        
        dl {
            display: flex;
            border-bottom: 1px solid var(--gray20);
            
            dt {
                width: 120px;
                background-color: var(--gray5);
                padding: 15px;
                font-weight: 600;
                color: var(--gray80);
                font-size: 1.4rem;
            }
            
            dd {
                flex: 1;
                padding: 15px;
                font-size: 1.4rem;
                color: var(--gray80);
            }
        }
    }
    
    .preview_info {
        background-color: var(--gray5);
        padding: 15px;
        border-radius: var(--radius-xs);
        
        dl {
            display: flex;
            margin-bottom: 5px;
            
            &:last-child {
                margin-bottom: 0;
            }
            
            dt {
                width: 120px;
                font-weight: 600;
                color: var(--gray70);
            }
            
            dd {
                color: var(--gray90);
            }
        }
    }
    
    .closeBtn {
        margin-top: 20px;
        text-align: center;
        
        button {
            padding: 10px 30px;
            background-color: var(--gray80);
            color: white;
            border: none;
            border-radius: var(--radius-xs);
            cursor: pointer;
        }
    }
  }
`;

const InstitutionSelect = ({ onNavigate, data }) => {
    const [selectedInstitution, setSelectedInstitution] = useState(null);
    const [isAccordionOpen, setIsAccordionOpen] = useState(true);
    const [showPreview, setShowPreview] = useState(false);
    const [searchCategory, setSearchCategory] = useState('cntrAdministDiv');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [aiResponse, setAiResponse] = useState(null);

    const autoRecommendations = [
        { id: '3200000', name: '서울특별시 관악구' },
        { id: '1210000', name: '국세청' },
        { id: '1320000', name: '경찰청' },
        { id: '6110000', name: '서울특별시' },
        { id: '3180000', name: '서울특별시 영등포구' },
    ];

    const manualInstitutions = [
        { id: '1000', name: '개인정보보호위원회' },
        { id: '1001', name: '경찰청' },
        { id: '1002', name: '고용노동부' },
        { id: '1003', name: '공정거래위원회' },
        { id: '1004', name: '과학기술정보통신부' },
        { id: '1005', name: '관세청' },
        { id: '1006', name: '교육부' },
        { id: '1007', name: '국가건축정책위원회' },
        { id: '1008', name: '국가교육위원회' },
        { id: '1009', name: '국가데이터처' },
        { id: '1010', name: '국가보훈부' },
        { id: '1011', name: '국가유산청' },
        { id: '1012', name: '국가인권위원회' },
        { id: '1013', name: '국무조정실' },
        { id: '1014', name: '국무총리비서실' },
        { id: '1015', name: '국민권익위원회' },
    ];

    const handleSelect = (name) => {
        setSelectedInstitution(name);
        setAiResponse(null); // Reset AI response when institution changes
    };

    const handlePreview = async () => {
        if (!selectedInstitution) {
            alert("처리기관을 선택해주세요.");
            return;
        }

        setIsSubmitting(true);
        try {
            const response = await fetch('http://localhost:8000/api/submit-proposal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            console.log("AI Analysis Result:", result);
            setAiResponse(result.final_answer);
            setShowPreview(true);
        } catch (error) {
            console.error("Error generating preview:", error);
            alert("미리보기 생성 중 오류가 발생했습니다.");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <FormWrapper>
            <SubTitle>
                <h3>일반제안 신청</h3>
            </SubTitle>

            <StepIndicator className="default_order">
                <div className="order_item completed">
                    <em>1</em> 신청서 작성
                </div>
                <div className="order_item on">
                    <em>2</em> 기관선택
                </div>
                <div className="order_item">
                    <em>3</em> 신청완료
                </div>
            </StepIndicator>

            <InfoBox>
                <div className="box_pt">
                    <p><strong>처리기관을 잘 모르는 경우</strong></p>
                    <p>※ 상담 방법 안내</p>
                    <ul>
                        <li>(전화) 110콜센터(국번없이 110)에 문의하여 적절한 처리기관을 안내 받으시기 바랍니다.</li>
                        <li>(인터넷) <a href="#" className="red">정부24(http://www.gov.kr)</a>에서 <b>처리기관별 주요업무</b>를 확인하실수 있습니다.</li>
                    </ul>
                </div>
            </InfoBox>

            <SectionTitle>
                <h4>처리기관 자동 추천</h4>
            </SectionTitle>

            <AutoRecommendationBox>
                <div className="kigwan_sel">
                    {autoRecommendations.map(inst => (
                        <button key={inst.id} type="button" onClick={() => handleSelect(inst.name)}>
                            <span>{inst.name}</span>
                        </button>
                    ))}
                </div>
                <p className="important_txt">※ 처리기관 자동추천 기능은 귀하께서 작성하신 내용(제목, 내용, 첨부파일을)을 분석하여 시스템적으로 적절한 기관들을 추천하는 서비스입니다.</p>
            </AutoRecommendationBox>

            <SectionTitle>
                <h4>처리기관 직접 선택</h4>
            </SectionTitle>

            <ManualSelectionBox>
                <a href="#" className={`slideBtn ${!isAccordionOpen ? 'closed' : ''}`} onClick={(e) => { e.preventDefault(); setIsAccordionOpen(!isAccordionOpen); }}>
                    {isAccordionOpen ? '상세내용 접기' : '상세내용 펼치기'}
                </a>

                {isAccordionOpen && (
                    <div className="auto_box">
                        <div className="value_smallbox">
                            <strong>기관분류</strong>
                            <span className="radioBtn">
                                <input type="radio" id="value_01" name="value_d" checked={searchCategory === 'cntrAdministDiv'} onChange={() => setSearchCategory('cntrAdministDiv')} />
                                <label htmlFor="value_01">중앙행정기관</label>
                            </span>
                            <span className="radioBtn">
                                <input type="radio" id="value_02" name="value_d" checked={searchCategory === 'locgovDiv'} onChange={() => setSearchCategory('locgovDiv')} />
                                <label htmlFor="value_02">지방자치단체</label>
                            </span>
                            <span className="radioBtn">
                                <input type="radio" id="value_03" name="value_d" checked={searchCategory === 'eduOrgDiv'} onChange={() => setSearchCategory('eduOrgDiv')} />
                                <label htmlFor="value_03">교육청</label>
                            </span>
                            <span className="radioBtn">
                                <input type="radio" id="value_04" name="value_d" checked={searchCategory === 'pblinstDiv'} onChange={() => setSearchCategory('pblinstDiv')} />
                                <label htmlFor="value_04">공공기관</label>
                            </span>
                            <span className="radioBtn">
                                <input type="radio" id="value_06" name="value_d" checked={searchCategory === 'searchInst'} onChange={() => setSearchCategory('searchInst')} />
                                <label htmlFor="value_06">기관검색</label>
                            </span>
                        </div>

                        {searchCategory === 'searchInst' && (
                            <div className="schBox">
                                <div className="schRow">
                                    <label htmlFor="instSearchWord">기관 검색</label>
                                    <div className="sch_btn_right">
                                        <input type="text" id="instSearchWord" />
                                        <button type="button">검색</button>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className="inner_bd">
                            {manualInstitutions.map(inst => (
                                <button
                                    key={inst.id}
                                    type="button"
                                    className={selectedInstitution === inst.name ? 'on' : ''}
                                    onClick={() => handleSelect(inst.name)}
                                >
                                    <span>{inst.name}</span>
                                </button>
                            ))}
                        </div>

                        <div className="auto_more">
                            <button><span>기관 더 보기</span></button>
                        </div>
                    </div>
                )}
            </ManualSelectionBox>

            {selectedInstitution && (
                <ConfirmationBox>
                    <strong>강영호 님의 제안을 <span>"{selectedInstitution}"</span>(으)로 신청하시겠습니까?</strong>
                </ConfirmationBox>
            )}

            <ButtonArea className="btnAreaLR">
                <div className="btnA_l">
                    <button className="btn gray" onClick={() => onNavigate('form')}>이전</button>
                    <button className="btn gray">취소</button>
                </div>
                <div className="btnA_r">
                    <button
                        className="btn line"
                        onClick={handlePreview}
                        disabled={isSubmitting}
                    >
                        {isSubmitting ? '생성중...' : '미리보기'}
                    </button>
                    {selectedInstitution && <button className="btn fill">신청</button>}
                </div>
            </ButtonArea>

            {showPreview && (
                <ModalOverlay>
                    <ModalContent>
                        <div className="layerPop_top">
                            <strong>미리보기</strong>
                            <button onClick={() => setShowPreview(false)}>닫기</button>
                        </div>
                        <div className="def_lPop_body">
                            <p>작성하신 내용에 대해 확인 후 신청해주시기 바랍니다.</p>
                            <div className="titBox">
                                <strong>{data?.prplTitl || '제목 없음'}</strong>
                            </div>
                            <div className="preview">
                                <dl>
                                    <dt>현황 및 문제점</dt>
                                    <dd>{data?.prplCntnCl || '-'}</dd>
                                </dl>
                                <dl>
                                    <dt>개선방안</dt>
                                    <dd>{data?.btmtIdeaCl || '-'}</dd>
                                </dl>
                                <dl>
                                    <dt>기대효과</dt>
                                    <dd>{data?.expcEfctCl || '-'}</dd>
                                </dl>
                            </div>

                            <div className="preview_info">
                                <dl>
                                    <dt>신청 기관</dt>
                                    <dd>{selectedInstitution || '-'}</dd>
                                </dl>
                            </div>

                            {aiResponse && (
                                <div className="preview" style={{ marginTop: '20px', borderTop: '2px solid var(--primary50)' }}>
                                    <dl>
                                        <dt style={{ backgroundColor: 'var(--primary5)', color: 'var(--primary50)' }}>예상답변</dt>
                                        <dd style={{ whiteSpace: 'pre-wrap' }}>{aiResponse}</dd>
                                    </dl>
                                </div>
                            )}

                            <div className="closeBtn">
                                <button onClick={() => setShowPreview(false)}>닫기</button>
                            </div>
                        </div>
                    </ModalContent>
                </ModalOverlay>
            )}
        </FormWrapper>
    );
};

export default InstitutionSelect;
