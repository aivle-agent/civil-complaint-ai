import React, { useState, useRef } from 'react';
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
  }
`;

const WarningBox = styled.div`
  background-color: var(--gray5);
  padding: 20px;
  border-radius: var(--radius-sm);
  margin-bottom: 30px;
  
  ul {
    list-style: disc;
    padding-left: 20px;
    
    li {
      font-size: 1.4rem;
      color: var(--gray70);
      margin-bottom: 5px;
      
      &:last-child {
        margin-bottom: 0;
      }
    }
  }
`;

const FormTitle = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 15px;
  border-bottom: 2px solid var(--gray80);
  padding-bottom: 10px;

  h4 {
    font-size: 2rem;
    font-weight: 700;
    color: var(--gray90);
  }

  p {
    font-size: 1.3rem;
    color: var(--gray60);
    
    .red {
      color: var(--danger50);
      margin-right: 2px;
    }
  }
`;

const InputBox = styled.div`
  margin-bottom: 20px;
  border-bottom: 1px solid var(--gray10);
  padding-bottom: 20px;

  &:last-child {
    border-bottom: none;
  }

  .mwTit {
    display: block;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--gray90);
    margin-bottom: 10px;

    .red {
      color: var(--danger50);
      margin-left: 4px;
    }
  }

  input[type="text"] {
    width: 100%;
    height: 40px;
    border: 1px solid var(--gray30);
    padding: 0 10px;
    border-radius: var(--radius-xs);
    font-size: 1.5rem;

    &:focus {
      border-color: var(--primary50);
      outline: none;
    }
  }

  textarea {
    width: 100%;
    height: 200px;
    border: 1px solid var(--gray30);
    padding: 10px;
    border-radius: var(--radius-xs);
    font-size: 1.5rem;
    resize: vertical;

    &:focus {
      border-color: var(--primary50);
      outline: none;
    }
  }

  .charCnt {
    display: block;
    text-align: right;
    font-size: 1.3rem;
    color: var(--gray50);
    margin-top: 5px;
  }
`;

const FileUploadWrapper = styled.div`
  margin-top: 30px;
  
  .file_btn {
    display: inline-block;
    padding: 8px 16px;
    background-color: var(--gray80);
    color: white;
    border-radius: var(--radius-xs);
    cursor: pointer;
    font-size: 1.4rem;
    
    &:hover {
      background-color: var(--gray90);
    }
  }

  .addition_txt {
    font-size: 1.3rem;
    color: var(--gray60);
    margin-top: 10px;
  }

  .file_list {
    margin-top: 10px;
    border-top: 1px solid var(--gray20);
    
    li {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px;
      border-bottom: 1px solid var(--gray10);
      font-size: 1.4rem;
      color: var(--gray80);

      .delBtn {
        background: none;
        border: none;
        cursor: pointer;
        color: var(--danger50);
        font-size: 1.2rem;
        
        &:hover {
          text-decoration: underline;
        }
      }
    }
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
`;

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
`;

const ModalContent = styled.div`
  background: white;
  border-radius: var(--radius-md);
  width: 100%;
  max-width: 800px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  
  .modal-header {
    background: linear-gradient(135deg, var(--primary50) 0%, var(--primary60) 100%);
    padding: 20px 25px;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    
    h3 {
      color: white;
      font-size: 1.8rem;
      font-weight: 700;
      margin: 0;
    }
  }
  
  .modal-body {
    padding: 25px;
  }
  
  .section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary60);
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--primary50);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .original-section {
    background-color: var(--gray5);
    border-radius: var(--radius-sm);
    padding: 20px;
    margin-bottom: 25px;
    
    .field-group {
      margin-bottom: 15px;
      
      &:last-child {
        margin-bottom: 0;
      }
      
      label {
        display: block;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--gray70);
        margin-bottom: 5px;
      }
      
      p {
        font-size: 1.4rem;
        color: var(--gray80);
        background: white;
        padding: 12px;
        border-radius: var(--radius-xs);
        border: 1px solid var(--gray20);
        white-space: pre-wrap;
        line-height: 1.5;
      }
    }
  }
  
  .refined-section {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 2px solid var(--primary40);
    border-radius: var(--radius-sm);
    padding: 20px;
    
    p {
      font-size: 1.5rem;
      color: var(--gray90);
      line-height: 1.8;
      white-space: pre-wrap;
    }
  }
  
  .modal-footer {
    padding: 15px 25px;
    background-color: var(--gray5);
    border-radius: 0 0 var(--radius-md) var(--radius-md);
    display: flex;
    justify-content: flex-end;
    
    button {
      padding: 12px 30px;
      font-size: 1.5rem;
      font-weight: 600;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.2s;
      
      &.close-btn {
        background-color: var(--gray80);
        color: white;
        border: none;
        
        &:hover {
          background-color: var(--gray90);
        }
      }
    }
  }
`;

const ProposalForm = ({ onNext, data, onDataChange, aiResult, onAiResult }) => {
  // Local state for files is fine to keep here for now, or lift it if needed. 
  // The user specifically asked for text fields.
  const [files, setFiles] = useState([]);
  const fileInputRef = useRef(null);
  const [isRefining, setIsRefining] = useState(false);
  const [refinedResult, setRefinedResult] = useState(null);
  const [originalData, setOriginalData] = useState(null); // Store original data when refining

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    onDataChange(name, value);
  };

  const handleRefineQuestion = async () => {
    // Validate required fields first
    if (!data.prplTitl.trim() || !data.prplCntnCl.trim() || !data.btmtIdeaCl.trim() || !data.expcEfctCl.trim()) {
      alert("모든 필수 필드를 입력해주세요.");
      return;
    }

    // Save original data before refining
    setOriginalData({ ...data });
    setIsRefining(true);

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
      console.log("AI Refinement Result:", result);
      setRefinedResult(result);
      // 상위 컴포넌트에도 결과 전달 (InstitutionSelect에서 사용)
      if (onAiResult) {
        onAiResult(result);
      }
    } catch (error) {
      console.error("Error refining question:", error);
      alert("질문 교정 중 오류가 발생했습니다. 서버 연결을 확인해주세요.");
      setOriginalData(null);
    } finally {
      setIsRefining(false);
    }
  };

  const handleFileChange = (e) => {
    const newFiles = Array.from(e.target.files);
    // Simple validation (size check omitted for brevity, but structure is here)
    if (files.length + newFiles.length > 5) {
      alert("첨부파일은 최대 5개까지 등록 가능합니다.");
      return;
    }
    setFiles([...files, ...newFiles]);
  };

  const removeFile = (index) => {
    const newFiles = files.filter((_, i) => i !== index);
    setFiles(newFiles);
  };

  const handleNext = () => {
    if (!data.prplTitl.trim()) {
      alert("제안 제목을 입력해주세요.");
      document.getElementById('prplTitl').focus();
      return;
    }
    if (!data.prplCntnCl.trim()) {
      alert("현황 및 문제점을 입력해주세요.");
      document.getElementById('prplCntnCl').focus();
      return;
    }
    if (!data.btmtIdeaCl.trim()) {
      alert("개선방안을 입력해주세요.");
      document.getElementById('btmtIdeaCl').focus();
      return;
    }
    if (!data.expcEfctCl.trim()) {
      alert("기대효과를 입력해주세요.");
      document.getElementById('expcEfctCl').focus();
      return;
    }
    onNext();
  };

  return (
    <FormWrapper>
      <SubTitle>
        <h3>일반제안 신청</h3>
      </SubTitle>

      <StepIndicator className="default_order">
        <div className="order_item on">
          <em>1</em> 신청서 작성
        </div>
        <div className="order_item">
          <em>2</em> 기관선택
        </div>
        <div className="order_item">
          <em>3</em> 신청완료
        </div>
      </StepIndicator>

      <WarningBox className="bbBox">
        <ul>
          <li>제목과 내용은 접수 후 수정, 삭제가 불가능하므로 다시 확인하시고 신청해 주시기 바랍니다.</li>
          <li>로그인 유지시간(120분) 내 작성 완료하여 주시기 바랍니다.</li>
        </ul>
      </WarningBox>

      <FormTitle className="title_lr">
        <h4>제안 내용</h4>
        <p><span className="red">*</span> 표는 필수 입력사항입니다.</p>
      </FormTitle>

      <form id="frm">
        <InputBox className="mwInput_box">
          <strong className="mwTit">
            <label htmlFor="prplTitl">제안 제목<span className="red">*</span></label>
          </strong>
          <input
            type="text"
            name="prplTitl"
            id="prplTitl"
            value={data.prplTitl}
            onChange={handleInputChange}
            placeholder="제목을 입력하세요"
          />
          <span className="charCnt">({data.prplTitl.length}/200)</span>
        </InputBox>

        <InputBox className="mwInput_box">
          <strong className="mwTit">
            <label htmlFor="prplCntnCl">현황 및 문제점<span className="red">*</span></label>
          </strong>
          <textarea
            name="prplCntnCl"
            id="prplCntnCl"
            value={data.prplCntnCl}
            onChange={handleInputChange}
          />
          <span className="charCnt">({data.prplCntnCl.length}/4000)</span>
        </InputBox>

        <InputBox className="mwInput_box">
          <strong className="mwTit">
            <label htmlFor="btmtIdeaCl">개선방안<span className="red">*</span></label>
          </strong>
          <textarea
            name="btmtIdeaCl"
            id="btmtIdeaCl"
            value={data.btmtIdeaCl}
            onChange={handleInputChange}
          />
          <span className="charCnt">({data.btmtIdeaCl.length}/4000)</span>
        </InputBox>

        <InputBox className="mwInput_box">
          <strong className="mwTit">
            <label htmlFor="expcEfctCl">기대효과<span className="red">*</span></label>
          </strong>
          <textarea
            name="expcEfctCl"
            id="expcEfctCl"
            value={data.expcEfctCl}
            onChange={handleInputChange}
          />
          <span className="charCnt">({data.expcEfctCl.length}/4000)</span>
        </InputBox>

        <FileUploadWrapper className="fileItems">
          <strong className="mwTit">첨부파일</strong>
          <div className="item_input file">
            <label className="file_btn" htmlFor="progBoradWriteFile">
              파일 추가
            </label>
            <input
              type="file"
              id="progBoradWriteFile"
              className="hide"
              style={{ display: 'none' }}
              onChange={handleFileChange}
              multiple
              ref={fileInputRef}
            />
          </div>
          <p className="addition_txt">※ 첨부파일 최대 개수는 5개이며 첨부 가능 용량은 전체 90MB 입니다.</p>
          <ul className="file_list">
            {files.map((file, index) => (
              <li key={index}>
                <span>{file.name}</span>
                <button type="button" className="delBtn" onClick={() => removeFile(index)}>삭제</button>
              </li>
            ))}
          </ul>
        </FileUploadWrapper>
      </form>

      <ButtonArea className="btnAreaLR">
        <div className="btnA_l">
          <button className="btn gray">이전</button>
          <button className="btn gray">취소</button>
        </div>
        <div className="btnA_r">
          <button className="btn line">불러오기</button>
          <button
            className="btn line"
            onClick={handleRefineQuestion}
            disabled={isRefining}
          >
            {isRefining ? '교정 중...' : '질문 교정'}
          </button>
          <button className="btn fill" onClick={handleNext}>다음</button>
        </div>
      </ButtonArea>

      {/* Modal - shows when refinement is complete */}
      {refinedResult && originalData && (
        <ModalOverlay onClick={() => {
          setRefinedResult(null);
          setOriginalData(null);
        }}>
          <ModalContent onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>✨ 질문 교정 결과</h3>
            </div>

            <div className="modal-body">
              {/* Original Question Section */}
              <div className="section-title">📝 현재 질문 (원본)</div>
              <div className="original-section">
                <div className="field-group">
                  <label>제안 제목</label>
                  <p>{originalData.prplTitl}</p>
                </div>
                <div className="field-group">
                  <label>현황 및 문제점</label>
                  <p>{originalData.prplCntnCl}</p>
                </div>
                <div className="field-group">
                  <label>개선방안</label>
                  <p>{originalData.btmtIdeaCl}</p>
                </div>
                <div className="field-group">
                  <label>기대효과</label>
                  <p>{originalData.expcEfctCl}</p>
                </div>
              </div>

              {/* Refined Question Section */}
              <div className="section-title">🔄 AI 교정된 질문</div>
              <div className="refined-section">
                <p>{refinedResult.refined_question || '교정된 질문이 없습니다.'}</p>
              </div>
            </div>

            <div className="modal-footer">
              <button
                className="close-btn"
                onClick={() => {
                  setRefinedResult(null);
                  setOriginalData(null);
                }}
              >
                닫기
              </button>
            </div>
          </ModalContent>
        </ModalOverlay>
      )}
    </FormWrapper>
  );
};

export default ProposalForm;

