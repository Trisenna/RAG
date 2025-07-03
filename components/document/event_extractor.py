"""
聊天事件提取器实现
从聊天记录中提取结构化事件信息
"""
import os
import re
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.chat_models import ChatTongyi

from core.config.settings import settings
from core.exceptions.errors import DocumentProcessingException

logger = logging.getLogger(__name__)


class ChatEventExtractor:
    """聊天事件提取器"""

    def __init__(self, max_retries: int = 2):
        """
        初始化事件提取器

        Args:
            max_retries: LLM调用最大重试次数
        """
        self.max_retries = max_retries

        # 初始化LLM
        self.llm = None
        if settings.TONGYI_API_KEY:
            try:
                self.llm = ChatTongyi(
                    model_name="qwen-turbo",
                    temperature=0.1,
                    dashscope_api_key=settings.TONGYI_API_KEY
                )
                logger.info("聊天事件提取器LLM初始化成功")
            except Exception as e:
                logger.error(f"LLM初始化失败: {e}")
                raise DocumentProcessingException(
                    f"事件提取器LLM初始化失败: {str(e)}",
                    error_code="LLM_INIT_ERROR"
                )

    def is_chat_file(self, filename: str) -> bool:
        """
        判断是否为聊天文件

        Args:
            filename: 文件名

        Returns:
            是否为聊天文件
        """
        if not filename:
            return False

        # 移除文件扩展名
        name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename

        # 检查是否符合聊天文件命名模式
        chat_patterns = [
            r'.*_聊天$',  # 以_聊天结尾
            r'.*聊天记录.*',  # 包含聊天记录
            r'.*对话.*',  # 包含对话
            r'.*微信.*',  # 包含微信
            r'.*QQ.*',  # 包含QQ
        ]

        for pattern in chat_patterns:
            if re.match(pattern, name_without_ext, re.IGNORECASE):
                return True

        return False

    def extract_participants_from_filename(self, filename: str) -> List[str]:
        """
        从文件名中提取参与者信息

        Args:
            filename: 文件名

        Returns:
            参与者列表
        """
        participants = []

        # 移除文件扩展名和_聊天后缀
        name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
        name_clean = re.sub(r'_聊天$', '', name_without_ext)

        # 常见的分隔符
        separators = ['_', '-', '&', '和', '与', ',', '，']

        # 尝试按分隔符拆分
        for sep in separators:
            if sep in name_clean:
                parts = [part.strip() for part in name_clean.split(sep)]
                participants.extend([part for part in parts if part and len(part) <= 10])
                break
        else:
            # 如果没有分隔符，直接作为单个参与者
            if name_clean and len(name_clean) <= 10:
                participants.append(name_clean)

        return participants

    async def extract_events_async(self, chat_content: str, participants: List[str] = None) -> List[Dict[str, Any]]:
        """
        异步从聊天内容中提取事件

        Args:
            chat_content: 聊天内容
            participants: 参与者列表

        Returns:
            事件列表
        """
        if not self.llm:
            raise DocumentProcessingException(
                "LLM未初始化，无法进行事件提取",
                error_code="LLM_NOT_AVAILABLE"
            )

        for attempt in range(self.max_retries):
            try:
                prompt = self._build_event_extraction_prompt(chat_content, participants)
                response = await self.llm.apredict(prompt)
                events = self._parse_event_response(response)

                if events:
                    logger.debug(f"成功提取 {len(events)} 个事件")
                    return events

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"事件提取失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)

        logger.warning("事件提取最终失败，返回空列表")
        return []

    def extract_events_sync(self, chat_content: str, participants: List[str] = None) -> List[Dict[str, Any]]:
        """
        同步从聊天内容中提取事件

        Args:
            chat_content: 聊天内容
            participants: 参与者列表

        Returns:
            事件列表
        """
        if not self.llm:
            return []

        for attempt in range(self.max_retries):
            try:
                prompt = self._build_event_extraction_prompt(chat_content, participants)
                response = self.llm.predict(prompt)
                events = self._parse_event_response(response)

                if events:
                    return events

            except Exception as e:
                logger.warning(f"同步事件提取失败 (尝试 {attempt + 1}): {e}")

        return []

    def _build_event_extraction_prompt(self, chat_content: str, participants: List[str] = None) -> str:
        """构建事件提取提示"""
        participants_text = "、".join(participants) if participants else "对话参与者"

        return f"""
        你是一个专业的聊天记录分析师，请从以下聊天记录中提取具体的事件信息。

        聊天参与者：{participants_text}

        聊天内容：
        {chat_content}

        请识别并提取以下类型的事件：
        1. meeting_plan - 会议计划（包括工作会议、讨论等）
        2. meal_plan - 用餐计划（包括午餐、晚餐、聚餐等）
        3. travel_plan - 出行计划（包括交通安排、旅行等）
        4. shopping_plan - 购物计划（包括买东西、逛街等）
        5. entertainment_plan - 娱乐计划（包括看电影、游戏等）
        6. family_plan - 家庭计划（包括回家、探亲等）
        7. work_task - 工作任务（包括项目安排、任务分配等）
        8. social_event - 社交活动（包括聚会、约会等）
        9. other_plan - 其他计划

        对于每个事件，请提取以下信息：
        - event_type: 事件类型（从上面的类型中选择）
        - title: 事件标题（简短描述）
        - content: 事件详细内容
        - time: 事件时间（从聊天中提取的时间信息）
        - location: 地点（如果有的话）
        - participants: 参与者列表
        - status: 状态（计划中、已确认、已取消等）

        请以JSON数组格式返回，每个事件为一个JSON对象：

        [
          {{
            "event_type": "meal_plan",
            "title": "中午湘菜馆用餐",
            "content": "两人约定12点半去楼下新开的湘菜馆吃饭，李小雨推荐口水鸡",
            "time": "12点半",
            "location": "楼下湘菜馆",
            "participants": ["李小雨", "晓明"],
            "status": "已确认"
          }}
        ]

        注意：
        1. 只提取明确的、具体的事件，不要提取模糊的话题讨论
        2. 时间信息要尽可能具体
        3. 参与者要从实际的聊天记录中识别
        4. 如果没有明确的事件，返回空数组 []

        仅返回JSON数组，不要添加额外的文字说明。
        """

    def _parse_event_response(self, response: str) -> List[Dict[str, Any]]:
        """解析事件提取响应"""
        try:
            # 查找JSON数组
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                events = json.loads(json_str)

                if isinstance(events, list):
                    # 验证和清理事件数据
                    valid_events = []
                    for event in events:
                        if isinstance(event, dict) and self._validate_event(event):
                            valid_events.append(event)

                    return valid_events
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")

        return []

    def _validate_event(self, event: Dict[str, Any]) -> bool:
        """验证事件数据的有效性"""
        required_fields = ['event_type', 'title', 'content']

        # 检查必需字段
        for field in required_fields:
            if field not in event or not event[field]:
                return False

        # 验证事件类型
        valid_types = [
            'meeting_plan', 'meal_plan', 'travel_plan', 'shopping_plan',
            'entertainment_plan', 'family_plan', 'work_task', 'social_event', 'other_plan'
        ]

        if event['event_type'] not in valid_types:
            return False

        # 确保参与者是列表
        if 'participants' in event and not isinstance(event['participants'], list):
            event['participants'] = [str(event['participants'])] if event['participants'] else []

        return True

    def create_event_documents(self,
                               events: List[Dict[str, Any]],
                               source_doc: Document) -> List[Document]:
        """
        将事件转换为文档对象 - 修复版本

        Args:
            events: 事件列表
            source_doc: 源文档

        Returns:
            事件文档列表
        """
        event_docs = []

        # 获取正确的document_id和base_name
        document_id = source_doc.metadata.get("document_id", "")
        base_name = source_doc.metadata.get("base_name", "")
        filename = source_doc.metadata.get("filename", "")

        # 生成基础ID
        if base_name:
            base_id = base_name
        elif document_id:
            base_id = os.path.splitext(os.path.basename(document_id))[0]
        elif filename:
            base_id = os.path.splitext(filename)[0]
        else:
            base_id = "unknown"

        for i, event in enumerate(events):
            # 构建事件内容
            event_content = self._format_event_content(event)

            # 构建事件元数据 - 确保所有字段正确设置
            event_metadata = {
                **source_doc.metadata,  # 继承源文档的元数据
                "chunk_id": f"{base_id}_event_{i}",
                "chunk_index": i,
                "total_chunks": len(events),
                "document_id": document_id,  # 确保document_id正确

                # 明确设置布尔字段为Python布尔类型
                "is_event": True,
                "is_proposition": False,
                "is_backup_chunk": False,
                "semantic_level": 2,  # 事件级别

                # 事件特有字段
                "event_type": event.get('event_type', ''),
                "event_title": event.get('title', ''),
                "event_time": event.get('time', ''),
                "event_location": event.get('location', ''),
                "event_participants": event.get('participants', []),
                "event_status": event.get('status', ''),
            }

            # 调试输出
            logger.debug(f"创建事件文档 {i} ({event_metadata['chunk_id']}):")
            logger.debug(f"  is_event: {event_metadata['is_event']} (type: {type(event_metadata['is_event'])})")
            logger.debug(f"  document_id: {event_metadata['document_id']}")
            logger.debug(f"  base_id: {base_id}")
            logger.debug(f"  event_type: {event_metadata['event_type']}")

            # 创建事件文档
            event_doc = Document(
                page_content=event_content,
                metadata=event_metadata
            )

            event_docs.append(event_doc)

        logger.info(f"创建了 {len(event_docs)} 个事件文档，base_id: {base_id}")
        return event_docs

    def _format_event_content(self, event: Dict[str, Any]) -> str:
        """格式化事件内容为可读文本"""
        content_parts = [
            f"事件类型：{event.get('event_type', '')}",
            f"标题：{event.get('title', '')}",
            f"内容：{event.get('content', '')}",
        ]

        if event.get('time'):
            content_parts.append(f"时间：{event['time']}")

        if event.get('location'):
            content_parts.append(f"地点：{event['location']}")

        if event.get('participants'):
            participants_str = "、".join(event['participants'])
            content_parts.append(f"参与者：{participants_str}")

        if event.get('status'):
            content_parts.append(f"状态：{event['status']}")

        return "\n".join(content_parts)